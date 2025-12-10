from dataclasses import dataclass


@dataclass
class SanctumConfig:
    model_name: str = 'microsoft/Phi-3-mini-4k-instruct'
    load_in_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
