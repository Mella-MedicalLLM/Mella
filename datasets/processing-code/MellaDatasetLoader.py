import json
import pandas as pd
from datasets import load_dataset
import pathlib


class MellaDataset:
    def __init__(self, dataset_info):
        self.name_or_path = dataset_info["name_or_path"]
        self.options = dataset_info["options"]
        self.headers = dataset_info["headers"]
        self.type = dataset_info["type"]

        if self.type == "xml":
            if "xpath" not in dataset_info.keys():
                raise Exception("xml format need xpath ")
            self.xpath = dataset_info["xpath"]
        self.dataset = self.load_dataset()

    def _load_dataset_arguments(self):
        return [self.name_or_path] + self.options

    def load_dataset(self):
        if self.type == "hf":
            return load_dataset(*self._load_dataset_arguments())
        elif self.type == "xml":
            xml_file_paths = list(pathlib.Path(self.name_or_path).glob('**/*.xml'))
            dataset_key = "train"
            dataset_dict = {dataset_key: []}
            for xml_file_path in xml_file_paths:
                try:
                    piece_of_dataset = pd.read_xml(xml_file_path, xpath=self.xpath)[self.headers].values.tolist()
                    dataset_dict[dataset_key] = dataset_dict[dataset_key] + piece_of_dataset
                except:
                    pass
            return dataset_dict
        else:
            return None

    def save_csv(self, path):
        if self.dataset is None: return
        filename = self.name_or_path.split("/")[-1]
        for dataset_key in self.dataset.keys():
            dataframe = pd.DataFrame(self.dataset[dataset_key], columns=self.headers)
            dataframe[self.headers].to_csv(f"{path}/{filename}_{dataset_key}.csv", index=False)

    def __repr__(self):
        return f"{type(self).__name__}" + "{\n" + \
            f"\tname_or_path:{self.name_or_path},\n\t" + \
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
loader.save_csv("../default-dataset")