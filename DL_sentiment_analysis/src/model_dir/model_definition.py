import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids, attention_mask=None):
        # ids = [batch size, seq len]
        output = self.transformer(
            input_ids=ids,
            attention_mask=attention_mask,
        )
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction
