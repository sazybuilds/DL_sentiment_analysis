#Here I would load the model and then create the predict function
import torch
import transformers
from src.model_dir.model_definition import Transformer

transformer_name = "bert-base-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)#tokenizer initialized

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transformer = transformers.AutoModel.from_pretrained(transformer_name)


output_dim = len(train_data["label"].unique())
freeze = False


model = Transformer(transformer, output_dim, freeze)


model.load_state_dict(torch.load("model_dir/transformer.pt"))




def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return {
        "prediction": predicted_class,
        "probability": predicted_probability
    }
