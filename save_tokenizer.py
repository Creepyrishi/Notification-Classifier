from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
tokenizer.save_pretrained("./tokenizer")