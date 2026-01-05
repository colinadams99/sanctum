from pathlib import Path
import json
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



# sets  phi 3 model and sets  prompt paths
DEFAULT_MODEL = 'microsoft/Phi-3-mini-4k-instruct'
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / 'prompts'/ 'insight_prompt.txt'
FIN_PERSONA_PATH = Path(__file__).resolve().parents[1] / 'prompts' / 'finance_persona.txt'
FIN_TASK_PATH = Path(__file__).resolve().parents[1] / 'prompts' / 'finance_task_prompt.txt'


# initialize insights
class InsightInput(BaseModel):
    cycle_length_days: int | None = None
    pms_days: list[int] | None = None
    symptoms: list[str] | None = None
    mood_notes: list[str] | None = None
    sleep_hours: list[float] | None = None

class SanctumAI:
    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast = True,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = 'nf4',
            bnb_4bit_use_double_quant = True,
            bnb_4bit_compute_dtype = torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = {'': 0},  # forces the entire model onto GPU, change to auto instead if there may be no gpu (i.e. customer)
            quantization_config = quant_config,
        )

        self.model.eval()
        torch.set_float32_matmul_precision("high")

        self.prompt_tmpl = DEFAULT_PROMPT_PATH.read_text(encoding = 'utf-8')

        # setting up the persona
        self.persona = (Path(__file__).resolve().parents[1] / 'prompts' / 'finance_persona.txt').read_text()

        # finance specific templates
        if FIN_PERSONA_PATH.exists():
            self.finance_persona = FIN_PERSONA_PATH.read_text(encoding = 'utf-8')
        else:
            self.finance_persona = (
                'You are a portfolio finance assistant. '
                'Explain structures and documents clearly without giving investment advice.'
            )

        if FIN_TASK_PATH.exists():
            self.finance_task_tmpl = FIN_TASK_PATH.read_text(encoding = 'utf-8')

        else:
            self.finance_task_tmpl = (
                'Document type: {{DOC_TYPE}}\n'
                'Context: {{CONTEXT}}\n'
                'Task: {{TASK}}\n\n'
                'Document text:\n{{DOC_TEXT}}\n\n'
                'ASSISTANT:\n'
            )

        # ============= debugging here to see if gpu is being used effectively ================
        try:
            p = next(self.model.parameters())
            print('[SanctumAI] first param device:', p.device)
        except StopIteration:
            print('[SanctumAI] model has no parameters?')

        print('[SanctumAI] hf_device_map:', getattr(self.model, 'hf_device_map', None))

    # fxn to format user data
    def _format_user_data(self, dd: dict) -> str:
        # makes the text readable for the prompt
        parts = []
        # loops through items and appends parts together
        for k, v in dd.items():
            if v is None:
                continue
            parts.append(f'{k}: {v}')
        return '\n'.join(parts)


    @torch.inference_mode()
    def generate_insight(self, user_data: dict, max_new_tokens: int = 140) -> str:
        # validate & format
        dd = InsightInput(**user_data).model_dump()
        user_blob = self._format_user_data(dd)

        # builds the  prompt
        prompt = (
                self.persona
                + '\n\n'
                + self.prompt_tmpl.replace('{{USER_DATA}}', user_blob)
                + '\nASSISTANT:\n'
        )

        inputs = self.tokenizer(prompt, return_tensors = 'pt').to('cuda:0')


        out = self.model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            eos_token_id = self.tokenizer.eos_token_id,
        )

        # decodes the recent generated tokens
        gen = out[0][inputs['input_ids'].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens = True).strip()

    def _fill_finance_prompt(
            self,
            doc_type: str,
            task: str,
            doc_text: str,
            context: str | None = None,
    ) -> str:
        ctx = context or ''
        prompt = self.finance_task_tmpl
        prompt = prompt.replace('{{DOC_TYPE}}', doc_type)
        prompt = prompt.replace('{{TASK}}', task)
        prompt = prompt.replace('{{CONTEXT}}', ctx)
        prompt = prompt.replace('{{DOC_TEXT}}', doc_text)
        # returns the prompt but also treis to keep the output concise and following the requested format by the user.
        return (
                self.finance_persona
                + '\n\n'
                + prompt
                + '\n\nIMPORTANT: Keep the answer concise and stop after completing the requested format.'
        )

    @torch.inference_mode()
    def analyze_finance_doc(
            self,
            doc_text: str,
            doc_type: str = 'other',
            task: str = 'Summarize the key terms and structure for a portfolio finance analyst.',
            context: str | None = None,
            max_new_tokens: int = 120,
            temperature: float = 0.4,
            top_p: float = 0.9,
            do_sample: bool = True,
    ) -> str:
        prompt = self._fill_finance_prompt(
            doc_type = doc_type,
            task = task,
            doc_text = doc_text,
            context = context
        )



        toks = self.tokenizer(prompt, return_tensors = 'pt').to(self.model.device)
        out = self.model.generate(
            **toks,
            max_new_tokens = max_new_tokens,
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            eos_token_id = self.tokenizer.eos_token_id,
        )

        gen = out[0][toks['input_ids'].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens = True).strip()


