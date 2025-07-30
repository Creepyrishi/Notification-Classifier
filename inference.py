import torch
from classifier_head import ClassifierHead
from sentence_transformers import SentenceTransformer

embedding_model_name = "huawei-noah/TinyBERT_General_4L_312D"
embedding_model = SentenceTransformer(embedding_model_name)

head_model = ClassifierHead(312)
head_model.load_state_dict(torch.load('./Models/best_model.pth'))
head_model.eval()

def inference(embedding_model, head_model, input_sentence):
    classes = ["imp_no_urgent", "low", "moderate", "urgent"]

    embedding = embedding_model.encode(input_sentence)
    class_prediction = head_model(torch.tensor(embedding.reshape(1, 312), dtype=torch.float32))

    print(classes[int(torch.argmax(class_prediction))])

inference(embedding_model, head_model, "Your father died come soon")
