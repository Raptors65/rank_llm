import logging
import os
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import vllm
from ftfy import fix_text
from tqdm import tqdm

from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode
from rank_llm.rerank.vllm_handler import VllmHandler
from rank_llm.rerank.vllm_handler_with_openai_sdk import VllmHandlerWithOpenAISDK

from .listwise_rankllm import ListwiseRankLLM

logger = logging.getLogger(__name__)

ALPH_START_IDX = ord("A") - 1
TEMPLATES = files("rank_llm.rerank.prompt_templates")


class SweRankReranker(ListwiseRankLLM):
    """
    SweRankLLM reranker with support for both text and code (GitHub issue) reranking.
    
    Extends ListwiseRankLLM to add specialized handling for code localization tasks,
    particularly for ranking code functions based on their relevance to GitHub issues.
    """

    def __init__(
        self,
        model: str,
        name: str = "",
        context_size: int = 4096,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: Optional[str] = None,
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_gpus: int = 1,
        window_size: int = 20,
        stride: int = 10,
        batch_size: int = 32,
        base_url: Optional[str] = None,
        rerank_type: str = "text",
        code_prompt_type: str = "github_issue",
    ) -> None:
        """
        Creates instance of SweRankReranker for text or code reranking.

        Parameters:
            model (str): Identifier for the language model (e.g., Salesforce/SweRankLLM-Small).
            name (str, optional): Name for the model. Defaults to "".
            context_size (int, optional): Maximum tokens per prompt. Defaults to 4096.
            prompt_mode (PromptMode, optional): Deprecated. Use prompt_template_path.
            prompt_template_path (str, optional): Path to YAML prompt template.
            num_few_shot_examples (int, optional): Number of few-shot examples. Defaults to 0.
            few_shot_file (str, optional): Path to few-shot examples file.
            device (str, optional): Device for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
            num_gpus (int, optional): Number of GPUs to use. Defaults to 1.
            window_size (int, optional): Size of sliding window. Defaults to 20.
            stride (int, optional): Stride for sliding window. Defaults to 10.
            batch_size (int, optional): Batch size for processing. Defaults to 32.
            base_url (str, optional): Base URL for OpenAI-compatible API.
            rerank_type (str, optional): Type of reranking - "text" or "code". Defaults to "text".
            code_prompt_type (str, optional): Prompt type for code reranking. Defaults to "github_issue".
        """
        # Set default template based on rerank type if not provided
        if prompt_template_path is None:
            if rerank_type == "code":
                prompt_template_path = TEMPLATES / "swerank_github_issue_template.yaml"
            else:
                prompt_template_path = TEMPLATES / "swerank_text_template.yaml"

        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            window_size=window_size,
            stride=stride,
            use_alpha=False,  # SweRank uses numerical identifiers
            device=device,
            batch_size=batch_size,
        )

        self._name = name
        self._num_gpus = num_gpus
        self._base_url = base_url
        self._rerank_type = rerank_type
        self._code_prompt_type = code_prompt_type
        self._output_token_estimate = None

        # Validate rerank_type
        if rerank_type not in ["text", "code"]:
            raise ValueError(
                f"Invalid rerank_type: {rerank_type}. Must be 'text' or 'code'."
            )

        # Validate code_prompt_type for code reranking
        if rerank_type == "code" and code_prompt_type not in ["github_issue"]:
            raise ValueError(
                f"Invalid code_prompt_type: {code_prompt_type}. Must be 'github_issue'."
            )

        if self._device == "cuda":
            assert torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus

        # Initialize vLLM handler
        if self._base_url:
            self._vllm_handler = VllmHandlerWithOpenAISDK(
                model=model, base_url=base_url
            )
        else:
            self._vllm_handler = VllmHandler(
                model=model,
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=False,
                max_logprobs=30,
                tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.90,
                trust_remote_code=True,
            )
        self._tokenizer = self._vllm_handler.get_tokenizer()

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Reranks a batch of requests.
        """
        top_k_retrieve: int = kwargs.get("top_k_retrieve", rank_end)
        rank_end = min(top_k_retrieve, rank_end)
        populate_invocations_history: bool = kwargs.get(
            "populate_invocations_history", False
        )

        if (
            self._batch_size > 1
            and len(set([len(req.candidates) for req in requests])) != 1
        ):
            raise ValueError("Batched requests must have the same number of candidates")

        return self.sliding_windows_batched(
            requests,
            rank_start=max(rank_start, 0),
            rank_end=min(rank_end, len(requests[0].candidates)),
            top_k_retrieve=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            populate_invocations_history=populate_invocations_history,
        )

    def run_llm_batched(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        current_window_size: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """
        Run batched inference on multiple prompts.
        """
        if current_window_size is None:
            current_window_size = self._window_size

        logger.info("VLLM Generating!")
        max_tokens = self.num_output_tokens(current_window_size)

        if self._base_url:
            return self._vllm_handler.chat_completions(
                prompts=prompts, max_tokens=max_tokens, temperature=0
            )

        outputs = self._vllm_handler.generate_output(
            prompts=prompts,
            min_tokens=max_tokens,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return [
            (output.outputs[0].text, len(output.outputs[0].token_ids))
            for output in outputs
        ]

    def run_llm(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        current_window_size: Optional[int] = None,
    ) -> Tuple[str, int]:
        """
        Run inference on a single prompt.
        """
        if current_window_size is None:
            current_window_size = self._window_size

        max_tokens = self.num_output_tokens(current_window_size)

        if self._base_url:
            results = self._vllm_handler.chat_completions(
                prompts=[prompt], max_tokens=max_tokens, temperature=0
            )
            return results[0]

        output = self._vllm_handler.generate_output(
            prompts=[prompt],
            min_tokens=max_tokens,
            max_tokens=max_tokens,
            temperature=0.0,
        )[0]
        return output.outputs[0].text, len(output.outputs[0].token_ids)

    def create_prompt_batched(
        self,
        results: List[Result],
        rank_start: int,
        rank_end: int,
    ) -> List[Tuple[Union[str, List[Dict[str, str]]], int]]:
        """
        Create prompts for a batch of results.
        """

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_completed_prompts = []

        with ThreadPoolExecutor() as executor:
            for batch in chunks(results, self._batch_size):
                completed_prompts = list(
                    executor.map(
                        lambda result: self.create_prompt(result, rank_start, rank_end),
                        batch,
                    )
                )
                all_completed_prompts.extend(completed_prompts)
        return all_completed_prompts

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """
        Create a prompt for ranking using the inference handler.
        
        This method uses different document length limits based on rerank type:
        - Code: 1024 tokens (longer for code functions)
        - Text: 300 tokens (standard for passages)
        """
        query = result.query.text
        num = len(result.candidates[rank_start:rank_end])

        # Set max document length based on rerank type
        if self._rerank_type == "code":
            max_doc_length = 1024  # Longer for code
            min_doc_length = 300
        else:
            max_doc_length = 300  # Standard for text
            min_doc_length = 100

        max_query_len = self.get_num_tokens(query)

        while True:
            # Use the inference handler to create the prompt
            prompt_data = self._inference_handler.create_prompt(
                result=result,
                rank_start=rank_start,
                rank_end=rank_end,
                max_doc_length=max_doc_length,
                max_query_len=max_query_len,
            )

            if isinstance(prompt_data, tuple):
                prompt, num_tokens = prompt_data
            else:
                prompt = prompt_data
                num_tokens = self.get_num_tokens(prompt)

            # Check if prompt fits within context
            if num_tokens <= self.max_tokens() - self.num_output_tokens(
                rank_end - rank_start
            ):
                break
            else:
                # Truncate documents if prompt is too long
                prefix_len = len(
                    self._tokenizer.encode(
                        self._inference_handler.template["prefix"].format(
                            num=num, query=query
                        )
                    )
                )
                query_tokens = self._tokenizer.tokenize(query)

                if (len(query_tokens) + prefix_len) > (
                    self.max_tokens()
                    - min_doc_length * (rank_end - rank_start)
                    - self.num_output_tokens(rank_end - rank_start)
                ):
                    # Query truncation
                    offset = num_tokens - (
                        self.max_tokens() - self.num_output_tokens(rank_end - rank_start)
                    )
                    max_query_len -= offset // 2 + 1
                else:
                    # Document truncation
                    max_doc_length -= max(
                        1,
                        (
                            num_tokens
                            - self.max_tokens()
                            + self.num_output_tokens(rank_end - rank_start)
                        )
                        // ((rank_end - rank_start) * 4),
                    )

        return prompt, num_tokens

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Get the number of tokens in a prompt.
        """
        if isinstance(prompt, str):
            return len(self._tokenizer.encode(prompt))
        else:
            # For chat format
            text = self._tokenizer.apply_chat_template(prompt, tokenize=False)
            return len(self._tokenizer.encode(text))

    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Return cost per 1k tokens. Free for open-source models.
        """
        return 0.0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        """
        Estimate the number of output tokens for a given window size.
        """
        if current_window_size is None:
            current_window_size = self._window_size

        if self._output_token_estimate and self._window_size == current_window_size:
            return self._output_token_estimate

        # Estimate based on ranking output format: [1] > [2] > [3] > ...
        # For numerical identifiers only (SweRank doesn't use alpha)
        token_str = " > ".join([f"[{i+1}]" for i in range(current_window_size)])
        _output_token_estimate = len(self._tokenizer.encode(token_str)) + 2

        if self._window_size == current_window_size:
            self._output_token_estimate = _output_token_estimate

        return _output_token_estimate

    def get_output_filename(
        self,
        top_k_candidates: int,
        dataset_name: str,
        shuffle_candidates: bool,
        **kwargs: Any,
    ) -> str:
        """
        Generate output filename for reranking results.
        """
        _modelname = self._model.split("/")[-1]
        if _modelname.startswith("checkpoint"):
            _modelname = self._model.split("/")[-2] + "_" + _modelname

        name = f"{_modelname}_{self._context_size}_{top_k_candidates}"
        if dataset_name:
            name = f"{name}_{dataset_name}"
        if self._rerank_type == "code":
            name = f"{name}_code_{self._code_prompt_type}"
        if self._num_few_shot_examples > 0:
            name += f"_{self._num_few_shot_examples}_shot"

        from datetime import datetime

        return (
            f"{name}_shuffled_{datetime.isoformat(datetime.now())}"
            if shuffle_candidates
            else f"{name}_{datetime.isoformat(datetime.now())}"
        )

