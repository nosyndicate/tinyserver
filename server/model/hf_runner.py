from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from server.metrics.logging import log_event
from server.model.sampling import SamplingParams


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class ModelRunner:

    def __init__(self, config: ModelConfig):
        self.config = config
        log_event(
            "model_init_start",
            model=config.model_name_or_path,
            device=config.device,
            dtype=str(config.dtype),
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.dtype,
            device_map="auto" if config.device == "cuda" else None,
        ).to(
            config.device  # type: ignore[arg-type]
        )

        self.model.eval()  # Set the model to evaluation mode
        log_event("model_init_done", model=config.model_name_or_path)

    @torch.inference_mode()
    def generate_text(
        self, prompt: str, sampling_params: SamplingParams
    ) -> tuple[str, int, int]:

        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_params.seed)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.config.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        do_sample = sampling_params.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=sampling_params.max_new_tokens,
            temperature=max(sampling_params.temperature, 1e-5) if do_sample else None,
            top_p=sampling_params.top_p if do_sample else None,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        out_text = full_text
        if sampling_params.stops:
            out_text = _apply_stop_strings(full_text, sampling_params.stops)  # type: ignore[arg-type]

        output_tokens = int(outputs.shape[1]) - prompt_tokens
        return out_text, prompt_tokens, output_tokens  # type: ignore[return-value]


def _apply_stop_strings(text: str, stop_strings: list[str]) -> str:
    """Apply stop strings to the generated text."""
    cut = None
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]
