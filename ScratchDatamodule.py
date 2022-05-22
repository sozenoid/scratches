import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision import transforms
from ScratchDataset import ScratchDataset


class ScratchDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 2
        self.num_workers = 4
    def prepare_data(self):
        # download
        self.scratch_train = ScratchDataset(dir = '/home/macenrola/Hug_2Tb/DamageDetection/carScratchDetector/train')
        self.scratch_val = ScratchDataset(dir = '/home/macenrola/Hug_2Tb/DamageDetection/carScratchDetector/val')
        self.scratch_predict = ScratchDataset(dir = '/home/macenrola/Hug_2Tb/DamageDetection/carScratchDetector/val')

    def train_dataloader(self):
        return self._dataloader(self.scratch_train, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.scratch_val)


    def predict_dataloader(self):
        return self._dataloader(self.scratch_val)

    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )


def collate_fn(batch):
    return tuple(zip(*batch))
