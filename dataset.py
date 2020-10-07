from torch.utils.data import Dataset

class WikiDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.n_samples = len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]

    def __len__(self):
        return self.n_samples