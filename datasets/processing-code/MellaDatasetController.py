import csv
import json
import os
import pandas as pd

class DatasetController:
    base_dataset_path = "../processed-dataset"
    save_path = None
    dataset_header = ["text"]

    def __init__(self, save_path="../processed-dataset/combined",
                 max_row=None,
                 max_length=None,
                 keywords=None):
        self.dataset = None
        self.save_path = save_path
        self.max_length = max_length
        with open('dataset.json') as f:
            json_object = json.load(f)

        train_dataset = []
        validation_dataset = []
        test_dataset = []

        for dataset_info in json_object["dataset"]:
            if "name_or_path" in dataset_info.keys() is None:
                continue
            dataset_name = dataset_info["name_or_path"].split("/")[-1]
            entire_dataset = pd.DataFrame(columns=self.dataset_header)
            dataset_row_count = 0
            for dataset_type in ["train", "validation", "test"]:
                dataset_path = f"{self.base_dataset_path}/{dataset_name}_{dataset_type}.csv"
                if os.path.isfile(dataset_path):
                    dataset = pd.read_csv(dataset_path)
                    if max_row is not None and max_row < len(dataset):
                        dataset = dataset[:max_row]
                    dataset_row_count += len(dataset)
                    entire_dataset = pd.concat([entire_dataset, dataset])
            print(f"dataset_name: {dataset_name}: {dataset_row_count}")

            if max_length is not None:
                mask = entire_dataset.apply(lambda x: len(x[0]) < max_length, axis=1)
                entire_dataset = entire_dataset[mask]

            if keywords is not None:
                mask = entire_dataset.apply(lambda x: any(keyword.lower() in x[0].lower() for keyword in keywords),
                                            axis=1)
                entire_dataset = entire_dataset[mask]

            index1 = int(len(entire_dataset) * 0.8)
            index2 = int(len(entire_dataset) * 0.8) + int(len(entire_dataset) * 0.1)
            train = entire_dataset[:index1]
            validation = entire_dataset[index1:index2]
            test = entire_dataset[index2:]

            train_dataset.append(train)
            validation_dataset.append(validation)
            test_dataset.append(test)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def concat_datasets(self, tag=None):
        if tag is None:
            tag = f"{self.max_length}"
        self.concat_train_dataset(tag)
        self.concat_validation_dataset(tag)
        self.concat_test_dataset(tag)

    def concat_train_dataset(self, tag):
        train_dataset = pd.DataFrame(columns=self.dataset_header)
        for dataset in self.train_dataset:
            train_dataset = pd.concat([train_dataset, dataset])

        print("--------------------")
        print(f"train_dataset_{tag}_count:", train_dataset.count())
        print(f"train_{tag}_max_length:", train_dataset.apply(lambda x: len(x[0]), axis=1).agg(max))
        train_dataset.to_csv(f"{self.save_path}/train_dataset_{tag}.csv", index=False)

    def concat_validation_dataset(self, tag):
        validation_dataset = pd.DataFrame(columns=self.dataset_header)
        for dataset in self.validation_dataset:
            validation_dataset = pd.concat([validation_dataset, dataset])

        print("--------------------")
        print(f"validation_dataset_{tag}_count:", validation_dataset.count())
        print(f"validation_{tag}_max_length:", validation_dataset.apply(lambda x: len(x[0]), axis=1).agg(max))
        validation_dataset.to_csv(f"{self.save_path}/validation_dataset_{tag}.csv", index=False)

    def concat_test_dataset(self, tag):
        if tag is None:
            tag = self.max_length
        test_dataset = pd.DataFrame(columns=self.dataset_header)
        for dataset in self.test_dataset:
            test_dataset = pd.concat([test_dataset, dataset])
        print("--------------------")
        print(f"test_dataset_{tag}_count:", test_dataset.count())
        print(f"test_{tag}_max_length:", test_dataset.apply(lambda x: len(x[0]), axis=1).agg(max))
        test_dataset.to_csv(f"{self.save_path}/test_dataset_{tag}.csv", index=False)


print("### 1024")
DatasetController(max_row=100000, max_length=1024).concat_datasets()
print("\n\n")