## Dataset

### dataset.json
json file specifying the dataset to use  

**format**
```json
{
  "dataset": [
    {
      "name_or_path": "name or path",
      "options": ["hf-options"],
      "headers": ["dataset headers to use"],
      "type": "dataset type (hf or xml)",
      "xpath": "xpath to use if type is xml"
    }
  ]
}
```

### 1. DatasetLoader
[**MellaDatasetLoader**](./processing-code/MellaDatasetLoader.py)  
load & save datasets in datasets.json  
available: hf, xml(local)  


### 2. DatasetType
[**MellaDatasetType**](./processing-code/MellaDatasetType.py)

**DatasetType**  
Enum for linking data processor and files

**DatasetProcessor**  
Convert each dataset to a text format for fine-tuning  
- Each dataset processor class inherit DatasetProcessor
```csv
Question,Answer
A, B

<!-- convert -->

text
<s>[INST] A [/INST] B </s>
```

### 3. DatasetProcessingModel
[**MellaDatasetProcessingModel**](./processing-code/MellaDatasetProcessingModel.py)  
A model for storing files in one format using dataset processor 

### 4. DatasetController
[**MellaDatasetController**](./processing-code/MellaDatasetController.py)  
combine entire datasets & split train, validation, test dataset

