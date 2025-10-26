import torch
import torch.nn as nn

# Logistic Regression Model

class LogisticRegressionModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)

# Feedforward MLP

class FeedForwardMLP(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden_dims=[128, 64], num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        layers = []
        input_dim = d_model
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, num_classes), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)
        return self.network(x).squeeze(-1)

# Encoder Transformer

class BasicEncoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)

# Decoder Transformer

class BasicDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1).bool()
        x = self.decoder(x, x, tgt_mask=mask)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)
