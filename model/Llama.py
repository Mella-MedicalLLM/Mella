import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import (
    AutoTokenizer,
    pipeline
)


class Llama:
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = None
    pipeline = None
    device_map = {"": 0}

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map=self.device_map,
        )

    def text_generation(self, prompt: str):
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=256,
        )
        return sequences[0]['generated_text']