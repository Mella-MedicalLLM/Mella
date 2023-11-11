import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import pandas as pd

def split_instruction(text):
    for pt in [r"<s>\[INST\]", r"</s>"]:
        text = re.sub(pt, "", text)
    return re.split(r"\[/INST\]", text)[0].strip()


def split_example(text):
    for pt in [r"<s>\[INST\]", r"</s>"]:
        text = re.sub(pt, "", text)
    result = re.split(r"\[/INST\]", text)
    return result[1].strip() if len(result) > 1 else result[0].strip()


def model_validation(name, model, validation_path):
    df = pd.read_csv(validation_path)
    result = pd.DataFrame()
    result['prompt'] = df['text'][:20].apply(split_instruction)
    result['example'] = df['text'][:20].apply(split_example)
    result[name] = result['prompt'].apply(lambda prompt: split_example(model.text_generation(prompt)))
    result.to_csv(f"{name}_validation_result.csv")


dataset_path = "/home/jeonhui/mella/datasets/validation_dataset_1024.csv"
mella = Mella()
model_validation("mella", mella, dataset_path)

llama = Llama()
model_validation("llama", llama, dataset_path)
