import torch
from classifier_head import ClassifierHead
from sentence_transformers import SentenceTransformer
import torch.nn as nn

bert_model_name = "huawei-noah/TinyBERT_General_4L_312D"
bert_model = SentenceTransformer(bert_model_name)

tokenizer_bert = bert_model._first_module().tokenizer
transformer_bert = bert_model._first_module().auto_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


head_model = ClassifierHead(312)
head_model.load_state_dict(torch.load('./Models/best_model.pth'))


class CombinedModel(nn.Module):
    def __init__(self, tokenizer_bert, transformer_bert, head):
        super().__init__()

        self.tokenizer_bert = tokenizer_bert
        self.transformer_bert = transformer_bert
        self.head = head
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer_bert(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token is at position 0
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Pass to classifier head
        logits = self.head(cls_token)
        return logits

dummy_text = "Evacuate the premises, emergency crews en route."

tokens = tokenizer_bert(dummy_text, return_tensors="pt", padding=True, truncation=True)
tokens = {k: v.to(device) for k, v in tokens.items()}  # move tokens to same device

input_ids=tokens["input_ids"]
attention_mask=tokens["attention_mask"]

combined_model = CombinedModel(tokenizer_bert, transformer_bert, head_model).to(device)
combined_model.eval()
print(combined_model(input_ids, attention_mask))

torch.onnx.export(
    combined_model,
    (input_ids, attention_mask),
    "combined_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size"}
    },
    opset_version=14
)
print("Exported to combined_model.onnx")
