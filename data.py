from torch.utils.data import Dataset


class WikiDataset(Dataset):
    def __init__(self, src_path, tgt_path=None):
        self.src = self.open_file(src_path)[:100]
        if tgt_path is not None:
            self.tgt = self.open_file(tgt_path)[:100]
        else:
            self.tgt = None
        self.size = len(self.src)

    def __getitem__(self, index):
        if self.tgt is not None:
            return self.src[index], self.tgt[index]
        else:
            return self.src[index]

    def __len__(self):
        return self.size

    @staticmethod
    def open_file(file_path):
        data = []
        with open(file_path, 'r', encoding="utf-8") as f:
            sents = f.readlines()
            for s in sents:
                data.append(s.strip())
        return data


