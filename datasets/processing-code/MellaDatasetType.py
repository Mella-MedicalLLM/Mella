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
    MedQuAD = "MedQuAD"

    @staticmethod
    def getDatasetType(mellaDatasetTypeString: str):
        datatypeString = mellaDatasetTypeString.lower().split("/")[-1]
        if MellaDatasetType.MedicalDialog.value.lower() == datatypeString:
            return MellaDatasetType.MedicalDialog
        elif MellaDatasetType.Medmcqa.value.lower() == datatypeString:
            return MellaDatasetType.Medmcqa
        elif MellaDatasetType.MedQA.value.lower() == datatypeString:
            return MellaDatasetType.MedQA
        elif MellaDatasetType.MedQuAD.value.lower() == datatypeString:
            return MellaDatasetType.MedQuAD

    def getDatasetProcessor(self):
        if self == MellaDatasetType.MedicalDialog:
            return MedicalDialogDatasetProcessor()
        elif self == MellaDatasetType.Medmcqa:
            return MedmcqaDatasetProcessor()
        elif self == MellaDatasetType.MedQA:
            return MedQADatasetProcessor()
        elif self == MellaDatasetType.MedQuAD:
            return MedQuADDatasetProcessor()


class DatasetProcessor(metaclass=ABCMeta):
    # dataframe = pd.DataFrame(columns=["instruction", "input", "output", "text"])
    dataframe = pd.DataFrame(columns=["text"])

    @abstractmethod
    def transform(self, dataset):
        pass

    def toText(self, instruction, output):
        # prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        # if input != "" or input is None:
        #     return f"{prompt}### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"
        # return f"{prompt}### Instruction:\n{instruction}\n### Response:\n{output}"
        return f"<s>[INST] {instruction} [/INST] {output} </s>"

    def append(self, instruction, output):
        # data = pd.DataFrame({"instruction": [instruction], "input": [input], "output": [output],
        data = pd.DataFrame({"text": [self.toText(instruction, output)]})
        self.dataframe = pd.concat([self.dataframe, data])


class MedicalDialogDatasetProcessor(DatasetProcessor):
    def toDictString(self, x):
        patient = re.search(r"[\"\'][ ]*patient[ ]*:.*[\"\'],[ ]*[\"\'][ ]*doctor[ ]*:", x).group(0)
        patient = re.sub("[\"\'][ ]*patient[ ]*: *", "", patient)
        patient = re.sub("[\"\'],[ ]*[\"\'][ ]*doctor[ ]*:", "", patient).strip()
        doctor = re.search(r"[\"\'],[ ]*[\"\'][ ]*doctor[ ]*:.*", x).group(0)
        doctor = re.sub("[\"\'],[ ]*[\"\'][ ]*doctor[ ]*:", "", doctor).strip()
        doctor = re.sub(r"[\"\']]", "", doctor).strip()
        return {'patient': patient, 'doctor': doctor}

    def transform(self, dataset):
        for row in dataset["utterances"]:
            dialog = self.toDictString(row)
            instruction = dialog["patient"].strip()
            input = ""
            output = dialog["doctor"].strip()
            self.append(instruction, output)
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
                exp = f"Answer is {ans[1]}"
        except:
            print(f"except: {exp}")
        return exp

    def transform(self, dataset):
        for idx in dataset.index:
            exp = dataset.loc[idx, 'exp']
            cop = dataset.loc[idx, 'cop']
            options = dataset.loc[idx, ['opa', 'opb', 'opc', 'opd']]
            instruction = dataset.loc[idx, 'question'].strip()
            output = self.get_output(exp, cop, *options).strip()
            self.append(instruction, output)
        return self.dataframe


class MedQADatasetProcessor(DatasetProcessor):
    def transform(self, dataset):
        if dataset is None:
            return
        for idx in dataset.index:
            instruction = dataset.loc[idx, 'question'].strip()
            output = dataset.loc[idx, 'answer'].strip()
            self.append(instruction, output)
        return self.dataframe


class MedQuADDatasetProcessor(DatasetProcessor):
    def transform(self, dataset):
        if dataset is None:
            return
        for idx in dataset.index:
            instruction = dataset.loc[idx, 'Question'].strip()
            instruction = re.sub(r" *\?", "?", instruction)
            output = dataset.loc[idx, 'Answer'].strip()
            self.append(instruction, output)
        return self.dataframe
