from pathlib import Path
import json
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "insight_prompt.txt"

class InsightInput(BaseModel):
    cycle_length_days: int | None = None
    pms_days: list[int] | None = None
    symptoms: list[str] | None = None
    mood_notes: list[str] | None = None
    sleep_hours: list[float] | None = None

class SanctumAI:
    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            kwargs.update(dict(load_in_4bit=True, torch_dtype=torch.float16))
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        self.prompt_tmpl = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")

    def _format_user_data(self, dd: dict) -> str:
        # make a compact, readable text block for the prompt
        parts = []
        for k, v in dd.items():
            if v is None:
                continue
            parts.append(f"{k}: {v}")
        return "\n".join(parts)

    @torch.inference_mode()
    def generate_insight(self, user_data: dict, max_new_tokens: int = 140) -> str:
        # validate & format
        dd = InsightInput(**user_data).model_dump()
        user_blob = self._format_user_data(dd)
        prompt = self.prompt_tmpl.replace("{{USER_DATA}}", user_blob)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
