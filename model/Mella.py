import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel

class Mella:
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_name = "mella"
    base_model = None
    model = None

    def __init__(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(self.base_model, self.model_name)
        model = model.merge_and_unload()
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

    def text_generation(self, prompt: str, base_model: bool = False):
        model = self.model
        if base_model:
            model = self.base_model
        pipe = pipeline(task="text-generation", model=model, tokenizer=self.tokenizer, max_length=1024)
        result = pipe(f"<s>[INST]{prompt}[/INST]")
        return result[0]['generated_text']