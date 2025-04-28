import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import load_dataset,load_dataset_homepage
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=64, num_heads=4, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 300, embed_dim))  # [1, seq_len, embed_dim]

        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=128)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x) + self.position_embedding[:, :x.size(1), :]  
        x = x.permute(1, 0, 2)  
        out = self.transformer(x)  
        out = out.mean(dim=0) 
        return self.classifier(out)



def train_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def main():
    # X, Y, encoder = load_dataset("processed_file", window_size=300, stride=50)
    X, Y, encoder = load_dataset_homepage("processed_file", window_size=300, stride=50)
    X = X.astype(np.float32)
    Y = Y.astype(np.int64)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = TransformerClassifier(input_dim=3, num_classes=len(encoder.classes_))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds, true = train_model(model, train_loader, test_loader, device=device)

    print("Classification Report:")
    print(classification_report(true, preds, target_names=encoder.classes_))


if __name__ == "__main__":
    main()
