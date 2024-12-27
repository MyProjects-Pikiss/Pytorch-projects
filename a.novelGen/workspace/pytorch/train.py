import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from soynlp.tokenizer import LTokenizer

# 데이터셋 클래스 정의
class NovelDataset(Dataset):
    def __init__(self, data_dir, tokenizer=None):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
        self.data = []
        for path in self.file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                self.data.extend(f.readlines())

        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        tokens = self.tokenizer(line)
        return torch.tensor(tokens, dtype=torch.long)

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        output = self.transformer(src_emb, tgt_emb)
        return self.fc(output)

# 학습 함수 정의
def train_model(data_dir, model, criterion, optimizer, num_epochs):
    tokenizer = LTokenizer()
    dataset = NovelDataset(data_dir, tokenizer=tokenizer.tokenize)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            src = batch[:, :-1]
            tgt = batch[:, 1:]
            outputs = model(src, tgt)
            loss = criterion(outputs.view(-1, model.fc.out_features), tgt.view(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    vocab_size = 5000
    embed_size = 128
    num_heads = 8
    num_layers = 6
    max_len = 512
    num_epochs = 10
    data_dir = "/data/processed_data/"

    model = TransformerModel(vocab_size, embed_size, num_heads, num_layers, max_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(data_dir, model, criterion, optimizer, num_epochs)
    torch.save(model.state_dict(), "novel_generator.pth")
    print("Model saved as 'novel_generator.pth'")
