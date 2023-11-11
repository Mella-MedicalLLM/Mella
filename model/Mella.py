import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from peft import PeftModel

class Mella:
    device_map = {"": 0}
    base_model_name = "NousResearch/Llama-2-7b-chat-hf"
    model_name = "mella"
    base_model = None
    model = None

    def __init__(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )
        model = PeftModel.from_pretrained(self.base_model, self.model_name)
        model = model.merge_and_unload()
        self.model = model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            return_full_text=False,
        )

    def text_generation(self, prompt: str):
        prompt = f"<s>[INST] {prompt} [/INST]"
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=256,
        )
        return sequences[0]['generated_text']