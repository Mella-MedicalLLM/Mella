import torch
from peft import PeftConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import os

class Mella:
    _config = None
    device = None
    peft_model_id = None
    model = None
    _tokenizer = None
    _pipeline = None

    def __init__(self,
                 model_id: str = 'mella',
                 base_model_name_or_path: str | None = 'llama/llama-2-7b-hf',
                 device: str = 'auto',
                 is_local_model: bool = True):
        self._set_device(device)
        self._load_configure(model_id)
        if base_model_name_or_path is not None:
            self._change_base_base_model_name_or_path(base_model_name_or_path)
        self._load_model(is_local_model)
        self._load_tokenizer()
        self._load_pipeline()

    # --- Model Settings ---
    def _set_device(self,
                    device: str = 'auto'):
        usable_device = 'cpu'
        is_device_auto = device == 'auto'
        if (is_device_auto or device == 'mps') \
                and torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            usable_device = torch.device('mps')
        if (is_device_auto or device == 'cuda') \
                and torch.cuda.is_available():
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            usable_device = torch.device('cuda')
        self.device = usable_device

    def _load_configure(self,
                        model_id: str):
        self.model_id = model_id
        self._config = PeftConfig.from_pretrained(model_id)

    def _change_base_base_model_name_or_path(self, model_name_or_path):
        self._config.base_model_name_or_path = model_name_or_path

    def _load_model(self, is_local_model: bool = True):
        self.model = LlamaForCausalLM.from_pretrained(
            self._config.base_model_name_or_path,
            torch_dtype='auto',
            offload_folder="offload",
            offload_state_dict=True,
            local_files_only=is_local_model
        ).eval().to(self.device)

    def _load_tokenizer(self):
        self._tokenizer = LlamaTokenizer.from_pretrained(self._config.base_model_name_or_path)

    def _load_pipeline(self):
        self._pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            tokenizer=self._tokenizer,
            device= self.device
        )

    # --- text_generation ---
    def text_generation(self, input: str):
        sequences = self._pipeline(
            input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self._tokenizer.eos_token_id,
            max_length=200
        )
        return sequences[0]['generated_text']


mella = Mella(device='mps')
inputs = 'throat a bit sore and want to get a good imune booster, especially in light of the virus. please advise. have not been in contact with nyone with the virus.\n'
mella.text_generation(inputs)


