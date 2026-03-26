import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_len=10):
        self.seq_len = seq_len
        self.vocab_size = 50  # 0 to 49, since max label is 49
        
        # Read CSV
        self.data = pd.read_csv(csv_path)
        # Convert date to datetime and sort
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Extract features: d1 to d7 as integers
        self.features = self.data[['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']].values.astype(int)
        
    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        # Get sequence of seq_len consecutive time steps
        seq = self.features[idx:idx + self.seq_len]  # (seq_len, 7)
        # Get target: next time step
        target = self.features[idx + self.seq_len]  # (7,)
        
        # One-hot encode sequence
        seq_onehot = []
        for row in seq:
            row_onehot = F.one_hot(torch.tensor(row), num_classes=self.vocab_size)  # (7, 50)
            seq_onehot.append(row_onehot)
        input_tensor = torch.stack(seq_onehot)  # (seq_len, 7, 50)
        
        # One-hot encode target
        target_tensor = F.one_hot(torch.tensor(target), num_classes=self.vocab_size)  # (7, 50)
        
        return input_tensor, target_tensor

# Example usage
if __name__ == "__main__":
    dataset = TimeSeriesDataset('data/data_all_l649.csv', seq_len=10)
    print(f"Dataset length: {len(dataset)}")
    sample_input, sample_target = dataset[0]
    print(f"Input shape: {sample_input.shape}")  # Should be (10, 7, 50)
    print(f"Target shape: {sample_target.shape}")  # Should be (7, 50)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in dataloader:
        batch_input, batch_target = batch
        print(f"Batch input shape: {batch_input.shape}")  # (4, 10, 7, 50)
        print(f"Batch target shape: {batch_target.shape}")  # (4, 7, 50)
        break
