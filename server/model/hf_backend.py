import logging

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from server.executor.types import Sequence
from server.metrics.logging import log_event
from server.model.types import ModelBackend, ModelConfig

logger = logging.getLogger(__name__)


def bytes_to_gb(bytes_value: int) -> str:
    """Converts bytes to gigabytes using the 1024 base."""
    gb = bytes_value / (1024**3)
    return f"{gb:.2f} GB"


def _get_available_memory(memory_utilization: float) -> int:
    """Returns the available GPU memory in bytes."""
    free_mem, total_mem = torch.cuda.mem_get_info()
    total_free_mem = free_mem * memory_utilization
    peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    # reserve some room for peak memory usage during model execution
    available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
    logger.info(f"available_mem: {bytes_to_gb(available_mem)}")

    return available_mem


def qwen3_cache_allocator(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    memory_utilization: float,
    block_size: int,
    device: str,
) -> None:
    available_mem = _get_available_memory(memory_utilization)

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    default_dtype = torch.get_default_dtype()
    default_dtype_size = torch.tensor([], dtype=default_dtype).element_size()
    block_bytes = (
        2 * num_layers * block_size * num_kv_heads * head_dim * default_dtype_size
    )  # 2 for key and value
    num_available_kv_blocks = int(available_mem // block_bytes)

    if num_available_kv_blocks <= 0:
        raise MemoryError(
            f"Not enough memory for even one block of KV cache. Available memory: {bytes_to_gb(available_mem)}, "
            f"required memory for one block: {bytes_to_gb(block_bytes)}."
        )

    kv_cache = torch.zeros(
        2,
        num_layers,
        num_available_kv_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=device,
    )

    for i in range(num_layers):
        model.model.layers[i].self_attn.k_cache = kv_cache[0, i]
        model.model.layers[i].self_attn.v_cache = kv_cache[1, i]


allocator_by_name = {
    "Qwen/Qwen3-1.7B": qwen3_cache_allocator,
}


class HFBackend(ModelBackend):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def prefill_batch(self, sequences: list[Sequence]) -> None:
        raise NotImplementedError("HFBackend.prefill_batch is not implemented yet")

    def decode_batch(self, sequences: list[Sequence]) -> None:
        raise NotImplementedError("HFBackend.decode_batch is not implemented yet")

    def release(self) -> None:
        # Hugging Face models don't require explicit resource release, but if there were any,
        # such as clearing GPU cache, it could be done here.
        pass

    @staticmethod
    def load_model(model_config: ModelConfig) -> "HFBackend":
        log_event(
            "model_init_start",
            model=model_config.model_name_or_path,
            device=model_config.device,
            dtype=str(model_config.dtype),
        )
        config = AutoConfig.from_pretrained(model_config.model_name_or_path)

        # Optional: register attention hook here if needed
        # ALL_ATTENTION_FUNCTIONS["paged"] = paged_attention_forward
        # config.attention_implementation = "paged"

        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            config=config,
            dtype=model_config.dtype,
            device_map="auto" if model_config.device == "cuda" else None,
        )

        model.eval()
        log_event("model_init_done", model=model_config.model_name_or_path)

        if model_config.model_name_or_path in allocator_by_name:
            allocator = allocator_by_name[model_config.model_name_or_path]
            allocator(
                model,
                config,
                model_config.memory_utilization,
                model_config.block_size,
                model_config.device,
            )

        else:
            raise ValueError(f"Unsupported model: {model_config.model_name_or_path}")

        return HFBackend(model, tokenizer)
