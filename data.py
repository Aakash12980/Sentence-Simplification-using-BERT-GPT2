from torch.utils.data import Dataset
import pickle


class WikiDataset(Dataset):
    def __init__(self, src_path, tgt_path=None, ref_path=None, ref=False):
        self.src = self.open_file(src_path)
        self.ref = None
        self.tgt = None
        if tgt_path is not None:
            self.tgt = self.open_file(tgt_path)
        if ref_path is not None:
            self.ref = self.open_file(ref_path, ref)
        self.size = len(self.src)

    def __getitem__(self, index):
        if self.tgt is not None and self.ref is not None:
            return self.src[index], self.tgt[index], self.ref[index]
        elif self.tgt is not None:
            return self.src[index], self.tgt[index], None
        else:
            return self.src[index], None, None

    def __len__(self):
        return self.size

    @staticmethod
    def open_file(file_path, ref=False):
        data = []
        if ref:
            ref_data = pickle.load(open(file_path, 'rb'))
            return ref_data

        else:
            with open(file_path, 'r', encoding="utf8") as f:
                sents = f.readlines()
                for s in sents:
                    data.append(s.strip())
            return data


