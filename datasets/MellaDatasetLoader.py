import json
import pandas as pd
from datasets import load_dataset


class MellaDataset:
    def __init__(self, dataset_info):
        self.name = dataset_info["name"]
        self.options = dataset_info["options"]
        self.headers = dataset_info["headers"]
        self.datasets = self.load_dataset()

    def _load_dataset_arguments(self):
        return [self.name] + self.options

    def load_dataset(self):
         return load_dataset(*self._load_dataset_arguments())

    def save_csv(self, path):
        filename = self.name.split("/")[-1]
        for dataset_key in self.datasets.keys():
            dataframe = pd.DataFrame(self.datasets[dataset_key])
            dataframe[self.headers].to_csv(f"{path}/{filename}_{dataset_key}.csv", index=False)

    def __repr__(self):
        return f"{type(self).__name__}" + "{\n" + \
            f"\tname:{self.name},\n\t" + \
            f"options:{self.options}\n" + \
            "}"

class MellaDatasetLoader:
    def __init__(self):
        with open('dataset.json') as f:
            json_object = json.load(f)
        datasets = []
        for dataset_info in json_object["dataset"]:
            datasets.append(MellaDataset(dataset_info))
        self.datasets = datasets

    def save_csv(self, path):
        for dataset in self.datasets:
            dataset.save_csv(path)


loader = MellaDatasetLoader()
loader.save_csv("./origin")