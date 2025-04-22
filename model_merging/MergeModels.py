import torch
from transformers import AutoTokenizer, EsmForSequenceClassification, EsmModel

model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
modelClass = EsmForSequenceClassification.from_pretrained("./model_trained")

import torch.nn as nn
class EsmForRegression(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = EsmModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.regressor = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # Use [CLS] token embedding
        return self.regressor(pooled)

model_name2 = "facebook/esm2_t6_8M_UR50D"
modelReg = EsmForRegression(model_name2)
modelReg.load_state_dict(torch.load("./esm_regression_model.pt", weights_only=True, map_location=torch.device('cpu')))

torch.save({'class_state_dict': modelClass.state_dict(), 'reg_state_dict': modelReg.state_dict()}, "./eval_models.tar")