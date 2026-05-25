import torch

from server.metrics.logging import log_event


def bytes_to_gb(bytes_value: int) -> str:
    """Converts bytes to gigabytes using the 1024 base."""
    gb = bytes_value / (1024**3)
    return f"{gb:.2f} GB"


def get_available_memory(memory_utilization: float) -> float:
    """Returns the available GPU memory in bytes."""
    free_mem, total_mem = torch.cuda.mem_get_info()
    total_free_mem = free_mem * memory_utilization
    peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    # reserve some room for peak memory usage during model execution
    available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
    log_event(f"available_mem: {bytes_to_gb(available_mem)}")

    return available_mem
