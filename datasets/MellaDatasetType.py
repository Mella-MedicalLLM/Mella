from enum import Enum
from abc import ABCMeta, abstractmethod
import re

import pandas as pd


class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class MellaDatasetType(StrEnum):
    MedicalDialog = "medical_dialog"
    Medmcqa = "medmcqa"
    MedQA = "MedQA-USMLE-4-options"

    @staticmethod
    def getDatasetType(mellaDatasetTypeString: str):
        datatypeString = mellaDatasetTypeString.lower().split("/")[-1]
        if MellaDatasetType.MedicalDialog.value.lower() == datatypeString:
            return MellaDatasetType.MedicalDialog
        elif MellaDatasetType.Medmcqa.value.lower() == datatypeString:
            return MellaDatasetType.Medmcqa
        elif MellaDatasetType.MedQA.value.lower() == datatypeString:
            return MellaDatasetType.MedQA

    def getDatasetProcessor(self):
        if self == MellaDatasetType.MedicalDialog:
            return MedicalDialogDatasetProcessor()
        elif self == MellaDatasetType.Medmcqa:
            return MedmcqaDatasetProcessor()
        elif self == MellaDatasetType.MedQA:
            return MedQADatasetProcessor()


class DatasetProcessor(metaclass=ABCMeta):
    dataframe = pd.DataFrame(columns=["instruction", "input", "output", "text"])

    @abstractmethod
    def transform(self, dataset):
        pass

    def toText(self, instruction, input, output):
        prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        if input != "" or input is None:
            return f"{prompt}### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"
        return f"{prompt}### Instruction:\n{instruction}\n### Response:\n{output}"

    def append(self, instruction, input, output):
        data = pd.DataFrame({"instruction": [instruction], "input": [input], "output": [output],
                             "text": [self.toText(instruction, input, output)]})
        self.dataframe = pd.concat([self.dataframe, data])


class MedicalDialogDatasetProcessor(DatasetProcessor):
    def toDictString(self, x):
        changePair = [
            (r"\'patient:", "\"patient\":\""),
            (r"\"patient:", "\"patient\":\""),
            (r"\'doctor:", "\"doctor\":\""),
            (r"\"doctor:", "\"doctor\":\""),
            (r"\', \"d", "\", \"d"),
            (r"\', \"p", "\", \"p"),
            (r"\'}", "\"}"),
            (r"5\'9\"", "5feet 9inch")
        ]

        changeValues = ["symptoms", "virus", "hi", "compassionate", "walking pneumonia", "cruddy", "house clothes",
                        "public", "bisto", "people", "rescue", "preventive", "preventive", "common cold",
                        "talk-to-doctor.",
                        "those with mild disease may be de-isolated 14 days after symptom onset, while those with severe disease may be de-isolated 14 days after achieving clinical stability (e.g. once supplemental oxygen is discontinued)"]

        x = '{' + re.sub(r"\"\"", "\'", x).strip()[1:-1] + '}'
        for origin, new in changePair:
            x = re.sub(origin, new, x)

        for value in changeValues:
            x = re.sub(f"\"{value}\"", f"\'{value}\'", x)
        return eval(x)

    def transform(self, dataset):
        for row in dataset["utterances"]:
            dialog = self.toDictString(row)
            instruction = dialog["patient"].strip()
            input = ""
            output = dialog["doctor"].strip()
            self.append(instruction, input, output)
        return self.dataframe


class MedmcqaDatasetProcessor(DatasetProcessor):
    def get_output(self, exp, cop, opa, opb, opc, opd):
        options = [('a', opa), ('b', opb), ('c', opc), ('d', opd)]
        if exp is None:
            exp = ""
        try:
            for op, value in options:
                exp = re.sub(f"\({op}\)", value, exp)
                exp = re.sub(f"\'{op}\'", value, exp)
            if exp.find('Ans') == -1:
                ans = options[int(cop)]
                exp = f"Ans is {ans[1]}"
        except:
            print(f"except: {exp}")
        return exp

    def transform(self, dataset):
        for idx in dataset.index:
            exp = dataset.loc[idx, 'exp']
            cop = dataset.loc[idx, 'cop']
            options = dataset.loc[idx, ['opa', 'opb', 'opc', 'opd']]
            instruction = dataset.loc[idx, 'question'].strip()
            input = ""
            output = self.get_output(exp, cop, *options).strip()
            self.append(instruction, input, output)
        return self.dataframe


class MedQADatasetProcessor(DatasetProcessor):
    def transform(self, dataset):
        if dataset is None:
            return
        for idx in dataset.index:
            instruction = dataset.loc[idx, 'question'].strip()
            input = ""
            output =dataset.loc[idx, 'answer'].strip()
            self.append(instruction, input, output)
        return self.dataframe
