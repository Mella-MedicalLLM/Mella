import torch
from peft import PeftConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers

if torch.cuda.is_available() : device = torch.device('cuda')
elif torch.backends.mps.is_available() : device = torch.device('mps')
else : device=torch.device('cpu')
print(f'Using {device}')

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

peft_model_id = "mella"
config = PeftConfig.from_pretrained(peft_model_id)
config.base_model_name_or_path = 'llama/llama-2-7b-chat-hf'

model = LlamaForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype='auto',
    offload_folder="offload",
    offload_state_dict = True,
    local_files_only=True
)

# model = model.to(device)

tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    tokenizer=tokenizer,
    device = device
)

sequences = pipeline(
    'i’m absolutely worried sick about getting coronavirus and what it would mean for me if i got it, because of my diabetes. is there anything you can say that can help me in any way? my anxiety is so high. thank you.\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")