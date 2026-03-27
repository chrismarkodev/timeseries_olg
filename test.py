from data_loader import load_data, get_value_mapping, encode_data, decode_observation
from model import Seq2SeqTransformer
import torch
import argparse


def evaluate(index_offset, seq_len=10, embed_size=128, num_heads=8, num_layers=6, model_path='model.pth', data_path='data/data_all_l649.csv'):
    raw_data = load_data(data_path)
    value2idx, idx2value = get_value_mapping(raw_data)
    encoded_data = encode_data(raw_data, value2idx)

    vocab_size = len(value2idx) + 2
    start_token = len(value2idx) + 1

    if index_offset not in ('last', 'prev'):
        raise ValueError("index_offset must be 'last' or 'prev'")

    if index_offset == 'last':
        target_idx = len(encoded_data) - 1
    else:
        target_idx = len(encoded_data) - 2

    if target_idx - seq_len < 0:
        raise ValueError('Not enough data for seq_len with this index_offset')

    src_seq = encoded_data[target_idx - seq_len:target_idx]
    true_target = raw_data[target_idx]

    input_seq = []
    for obs in src_seq:
        input_seq.extend(obs)

    input_tensor = torch.tensor([input_seq], dtype=torch.long)

    model = Seq2SeqTransformer(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        pred = model.generate(input_tensor, max_len=7, start_token=start_token)

    pred_tokens = pred[0].tolist()
    pred_values = decode_observation(pred_tokens, idx2value)

    print(f"=== Evaluate: {index_offset} observation (target index {target_idx}) ===")
    print("Input window indexes:", list(range(target_idx - seq_len, target_idx)))
    print("True target:", true_target)
    print("Predicted (decoded):", pred_values)
    print("Predicted (tokens):", pred_tokens)
    print("Absolute errors:", [abs(a - b) for a, b in zip(true_target, pred_values)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test seq2seq predictor on last/prev observation')
    parser.add_argument('--offset', choices=['last', 'prev'], default='last', help='Use last or previous target obs')
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--data_path', type=str, default='data/data_all_l649.csv')
    args = parser.parse_args()

    evaluate(index_offset=args.offset, seq_len=args.seq_len, model_path=args.model_path, data_path=args.data_path)
