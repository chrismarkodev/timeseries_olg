import pandas as pd

def load_data(file_path, date_col='date', value_cols=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']):
    """
    Load the CSV data, sort by date, and return list of observations.
    Each observation is a list of 7 integers.
    """
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    data = df[value_cols].values.tolist()
    return data

def create_sequences(data, seq_len):
    """
    Create sequences of length seq_len * 7 tokens and corresponding targets.
    """
    sequences = []
    targets = []
    for i in range(len(data) - seq_len):
        seq = []
        for j in range(seq_len):
            seq.extend(data[i + j])
        target = data[i + seq_len]
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

def split_data(sequences, targets, train_ratio=0.7, val_ratio=0.2):
    """
    Split sequences and targets into train, val, test.
    """
    total = len(sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_seq = sequences[:train_end]
    train_tar = targets[:train_end]
    val_seq = sequences[train_end:val_end]
    val_tar = targets[train_end:val_end]
    test_seq = sequences[val_end:]
    test_tar = targets[val_end:]
    return (train_seq, train_tar), (val_seq, val_tar), (test_seq, test_tar)

if __name__ == "__main__":
    data = load_data('data/data_all_l649.csv')
    print("Data length:", len(data))
    print("First observation:", data[0])
    sequences, targets = create_sequences(data, 10)
    print("Sequences length:", len(sequences))
    print("First sequence length:", len(sequences[0]))
    print("First target:", targets[0])
