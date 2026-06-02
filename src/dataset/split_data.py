from typing import Tuple
from src.dataset.DataSet import SurvivalDataSet
from torch.utils.data import Subset, DataLoader
import numpy as np


def split_dataset_by_ids(dataset : SurvivalDataSet, sample_ids, train_ids, val_ids, test_ids):
    """
    Split a SurvivalDataSet into train/val/test by matching each row's Sample ID
    against pre-defined ID lists (loaded from splits.json), so the split is the
    SAME patients as the classic-models notebook.

    `sample_ids` must be aligned row-for-row with the dataset, e.g. the array
    returned by TorchPreprocessing.get_data_set(..., return_ids=True).
    """
    sample_ids = np.asarray(sample_ids).astype(str)

    def _select(ids):
        mask = np.isin(sample_ids, np.asarray(ids).astype(str))
        idx = np.where(mask)[0]
        return dataset.X[idx], dataset.durations[idx], dataset.events[idx]

    return _select(train_ids), _select(val_ids), _select(test_ids)


def create_dataloaders_train_val_test(train_data : Subset,
                                      val_data: Subset,
                                      test_data : Subset,
                                      batch_size : int = 16) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the data loaders
    """
    
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True)
    
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    return (train_loader, val_loader, test_loader)
    