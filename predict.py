from data_loader import load_data
from model import Seq2SeqTransformer
import torch

# Parameters (should match training)
file_path = 'data/data_all_l649.csv'
seq_len = 10
embed_size = 128
num_heads = 8
num_layers = 6
vocab_size = 51
model_path = 'model.pth'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = load_data(file_path)

# Take the last seq_len observations
last_seq = data[-seq_len:]

# Flatten into input sequence
input_seq = []
for obs in last_seq:
    input_seq.extend(obs)
input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

# Load model
model = Seq2SeqTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Predict
prediction = model.generate(input_tensor)
predicted_values = prediction[0].tolist()

print("Predicted next observation (d1 to bonus):", predicted_values)