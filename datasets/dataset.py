import numpy as np
import torch


class DefaultDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, timing=True):
        self.data = data
        self.labels = labels

        if timing:
            self.data = np.sign(self.data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float()
        y = torch.tensor(self.labels[index]).long()
        return x, y

    def __len__(self):
        return len(self.data)


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, timing=True):
        self.data = data
        self.labels = labels

        if timing:
            self.data = np.sign(self.data)

    def __getitem__(self, index):
        anchor = torch.from_numpy(self.data[index])

        label = self.labels[index]

        # Find a positive example unequal to anchor
        pos_idx = np.random.choice(np.where(self.labels == label)[0])
        while pos_idx == index:
            pos_idx = np.random.choice(np.where(self.labels == label)[0])

        pos = torch.from_numpy(self.data[pos_idx])

        # Find a negative example
        neg_idx = np.random.choice(np.where(self.labels != self.labels[index])[0])
        neg = torch.from_numpy(self.data[neg_idx])

        # correctness check
        assert self.labels[pos_idx] == self.labels[index], "Positive example has different label"
        assert self.labels[neg_idx] != self.labels[index], "Negative example has same label"
        assert pos_idx != index, "Positive example is the same as anchor"

        return anchor, pos, neg

    def __len__(self):
        return len(self.data)
