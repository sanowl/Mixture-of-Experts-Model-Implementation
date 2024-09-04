import torch
from torch.utils.data import Dataset
from typing import Tuple

class SequenceDataset(Dataset):
    def __init__(self, num_samples: int, seq_length: int, input_size: int, num_classes: int):
        self.X = torch.randn(num_samples, seq_length, input_size)
        self.y = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

