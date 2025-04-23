import sys

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel
from typing import List

CLASS_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
REG_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def load_model(model_path: str) -> List[nn.Module]:    
	# prepare needed classes
    
    class EsmForRegression(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.backbone = EsmModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
            self.regressor = torch.nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask=None):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            return self.regressor(pooled)

    
    # load the data
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        modelClass = EsmForSequenceClassification.from_pretrained(CLASS_MODEL_NAME)
        modelReg = EsmForRegression(REG_MODEL_NAME)
        modelClass.load_state_dict(checkpoint['class_state_dict'])
        modelReg.load_state_dict(checkpoint['reg_state_dict'])
        return [modelClass, modelReg]
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def evaluateClass(model: nn.Module, data: pd.DataFrame):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    BATCH_SIZE = 32
    tokenizer = AutoTokenizer.from_pretrained(CLASS_MODEL_NAME)
    
    X = list(data["sequence"])
    X = tokenizer(X, return_tensors="pt")
    
    model.eval()
    predClass = []
    with torch.inference_mode():
        for i in range(0, len(X["input_ids"]), BATCH_SIZE):
            input_ids = X["input_ids"][i:i+BATCH_SIZE].to(device)
            attention_mask = X["attention_mask"][i:i+BATCH_SIZE].to(device)
            test_batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            y_logits_test = model(**test_batch).logits
            predClass = predClass + torch.argmax(y_logits_test, dim=1).tolist()
    
    return predClass


def evaluateReg(model: nn.Module, data: pd.DataFrame):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    BATCH_SIZE = 32
    tokenizer = AutoTokenizer.from_pretrained(REG_MODEL_NAME)
    
    class SequenceRegressionDataset(Dataset):
        def __init__(self, sequences, tokenizer):
            self.encodings = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
            self.length = len(sequences)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            item = {key: val[idx].to(device) for key, val in self.encodings.items()}
            return item

    X = list(data["sequence"])
    dataset = SequenceRegressionDataset(X, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    predReg = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        predReg = predReg + outputs.tolist()
        
    return predReg


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluation_script.py <path_to_model> <path_to_test_data>")
        sys.exit(1)

    model_path = sys.argv[1]
    path_test = sys.argv[2]

    modelClass, modelReg = load_model(model_path)
    data = pd.read_csv(path_test, delimiter='\t')
    data = data[:100]
    classPredictions = evaluateClass(modelClass, data)
    regPredictions = evaluateReg(modelReg, data)
    print("id\tpredicted_is_active\tpredicted_rna_dna_ratio")
    for i, cp, rp in zip(range(len(data)), classPredictions, regPredictions):
        print(f"{i}\t{cp}\t{rp[0]}")