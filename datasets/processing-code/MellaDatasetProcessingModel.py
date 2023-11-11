import json
import pandas as pd
import os.path
from MellaDatasetType import *


class DataProcessingModel:
    dataset_keys = ["train", "validation", "test"]
    dataset_names = []
    origin_dataset = {}
    processed_dataset = {}

    def load_dataset(self, dataset_path):
        if not os.path.isfile(dataset_path):
            return None
        dataset = pd.read_csv(dataset_path)
        return dataset.fillna("")

    def all_data_processing(self, path):
        with open('dataset.json') as f:
            json_object = json.load(f)
        self.dataset_names = list(map(lambda x: x["name_or_path"].split("/")[-1], json_object["dataset"]))
        for dataset_name in self.dataset_names:
            self.data_processing(path, dataset_name)

    def data_processing(self, path, dataset_name):
        dataset_type = MellaDatasetType.getDatasetType(dataset_name)
        data_processor = dataset_type.getDatasetProcessor()
        for dataset_key in self.dataset_keys:
            dataset_file_name = f"{dataset_name}_{dataset_key}"
            dataset_path = f"{path}/{dataset_file_name}.csv"
            self.origin_dataset[dataset_file_name] = self.load_dataset(dataset_path)
            self.processed_dataset[dataset_file_name] = data_processor.transform(self.origin_dataset[dataset_file_name])

    def save_all_processed_dataset(self, path):
        for dataset_file_name in self.processed_dataset.keys():
            self.save_processed_dataset(path, dataset_file_name)

    def save_processed_dataset(self, path, dataset_file_name):
        if self.processed_dataset[dataset_file_name] is None:
            return
        self.processed_dataset[dataset_file_name].to_csv(f"{path}/{dataset_file_name}.csv", index=False)


model = DataProcessingModel()
model.all_data_processing("../default-dataset")
model.save_all_processed_dataset("../processed-dataset")