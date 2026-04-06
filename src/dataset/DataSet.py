from torch.utils.data import Dataset
import torch


class SurvivalDataSet(Dataset):
    def __init__(self, X, durations, events) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.durations = torch.tensor(durations, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, indx):
        return self.X[indx], self.durations[indx], self.events[indx]