from transformers import LlamaTokenizerFast, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

PATH = "./llama/llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(PATH, local_files_only=True)
model = LlamaForCausalLM.from_pretrained(PATH, local_files_only=True)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer
)

sequences = pipeline(
    'i’m absolutely worried sick about getting coronavirus and what it would mean for me if i got it, because of my diabetes. is there anything you can say that can help me in any way? my anxiety is so high. thank you.\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")