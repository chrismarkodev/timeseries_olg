from data_loader import load_data, get_value_mapping, encode_data, decode_observation
from model import Seq2SeqTransformer
import torch

# Parameters (should match training)
file_path = 'data/data_all_l649.csv'
seq_len = 10
embed_size = 128
num_heads = 8
num_layers = 6
model_path = 'model.pth'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
raw_data = load_data(file_path)
value2idx, idx2value = get_value_mapping(raw_data)
encoded_data = encode_data(raw_data, value2idx)

vocab_size = len(value2idx) + 2
start_token = len(value2idx) + 1

# Take the last seq_len observations
last_seq = encoded_data[-seq_len:]

# Flatten into input sequence
input_seq = []
for obs in last_seq:
    input_seq.extend(obs)
input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

# Load model
model = Seq2SeqTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Predict
prediction = model.generate(input_tensor, max_len=7, start_token=start_token)
predicted_tokens = prediction[0].tolist()
predicted_values = decode_observation(predicted_tokens, idx2value)

print("Predicted next observation (d1..d7):", predicted_values)
print("Predicted token ids:", predicted_tokens)