import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

text = "70% off one your first purchase"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
session = ort.InferenceSession('./Models/combined_model.onnx', providers=['CPUExecutionProvider'])

tokens = tokenizer(text, return_tensors = "pt", padding = True, truncation=True)

# we need to see how it was exported
onnx_inputs  = {
    "input_ids" : tokens["input_ids"].numpy(), 
    "attention_mask": tokens["attention_mask"].numpy()
}

output = session.run(None, onnx_inputs)

print(output[0]) ##logits
