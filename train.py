from data_loader import load_data, create_sequences, split_data, get_value_mapping, encode_data
from model import Seq2SeqTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Parameters
file_path = 'data/data_all_l649.csv'
seq_len = 10
embed_size = 128
num_heads = 8
num_layers = 6
batch_size = 32
epochs = 100
lr = 1e-4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load data
raw_data = load_data(file_path)
value2idx, idx2value = get_value_mapping(raw_data)
encoded_data = encode_data(raw_data, value2idx)

vocab_size = len(value2idx) + 2  # data tokens + start token reserved
start_token = len(value2idx) + 1

sequences, targets = create_sequences(encoded_data, seq_len)
(train_seq, train_tar), (val_seq, val_tar), (test_seq, test_tar) = split_data(sequences, targets)

# Convert to tensors
train_seq = torch.tensor(train_seq, dtype=torch.long)
train_tar = torch.tensor(train_tar, dtype=torch.long)
val_seq = torch.tensor(val_seq, dtype=torch.long)
val_tar = torch.tensor(val_tar, dtype=torch.long)

# Prepare tgt_input and tgt_output for training
train_tgt_in = torch.cat([torch.full((train_tar.size(0), 1), start_token, dtype=torch.long), train_tar[:, :-1]], dim=1)
train_tgt_out = train_tar
val_tgt_in = torch.cat([torch.full((val_tar.size(0), 1), start_token, dtype=torch.long), val_tar[:, :-1]], dim=1)
val_tgt_out = val_tar

# DataLoader
train_dataset = TensorDataset(train_seq, train_tgt_in, train_tgt_out)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = Seq2SeqTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for src, tgt_in, tgt_out in train_dataloader:
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        optimizer.zero_grad()
        output = model(src, tgt_in)
        loss = criterion(output.view(-1, vocab_size), tgt_out.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_seq = val_seq.to(device)
        val_tgt_in = val_tgt_in.to(device)
        val_tgt_out = val_tgt_out.to(device)
        val_output = model(val_seq, val_tgt_in)
        val_loss = criterion(val_output.view(-1, vocab_size), val_tgt_out.view(-1))
        print(f'Val Loss: {val_loss.item()}')

# Save model
torch.save(model.state_dict(), 'model.pth')