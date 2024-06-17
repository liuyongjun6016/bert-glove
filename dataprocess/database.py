import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from typing import List, Dict

class CustomDataset(Dataset):
    def __init__(self, df: List[str], session_dict, nid2index: Dict[str, int]):
        self.data = df
        self.session_dict = session_dict
        self.nid2index = nid2index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = json.loads(self.data[idx])
        uid = row["uid"]

        if uid not in self.session_dict:
            return []

        session_index = self.session_dict[uid]

        if len(session_index) > 5:
            session_index = session_index[-5:]
        else:
            session_index = session_index + [0] * (5 - len(session_index))

        samples = []



        for pair in row["pairs"]:
            news_id = pair[0]
            pos_index = [self.nid2index[news_id]]
            random_neg_ids = pair[1]
            neg_len = len(random_neg_ids)
            neg_index = [self.nid2index[nid] for nid in random_neg_ids]
            sample_index = pos_index + neg_index

            # session_index = torch.LongTensor(session_index)
            # sample_index = torch.LongTensor(sample_index)
            # labels = torch.LongTensor([0,])

            labels = [1] + [0]*neg_len

            samples.append((session_index, sample_index, labels))

        return samples


def collate_fn(batch):
    # Remove None values (those that didn't have a matching uid)
    # batch = [item for sublist in batch for item in sublist if item is not None]
    # batch = [item for sublist in batch for item in sublist if item]
    # session_indices, sample_indices, labels = zip(*batch)
    #
    # return session_indices, sample_indices, labels

    session_indices = []
    sample_indices = []
    labels = []

    for item in batch:
        if item:  # 确保非空
            for session_index, sample_index, label in item:
                session_indices.append(session_index)
                sample_indices.append(sample_index)
                labels.append(label)

    return (
        torch.tensor(session_indices, dtype=torch.long),
        torch.tensor(sample_indices, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float),
    )

class ValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, session_dict, nid2index: Dict[str, int]):
        self.data = df[["uid", "hist", "imp", "id"]]
        self.session_dict = session_dict
        self.nid2index = nid2index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        uid = row[0]

        if uid not in self.session_dict:
            return []

        session_index = self.session_dict[uid]

        if len(session_index) > 5:
            session_index = session_index[-5:]
        else:
            session_index = session_index + [0] * (5 - len(session_index))

        samples = []

        for sample in row[2].strip().split():
            news_id, label = sample.split("-")[:2]

            if news_id not in self.nid2index:
                continue

            val_sample = [self.nid2index[news_id]]
            label = [int(label)]
            imp_id = [int(row[-1])]

            samples.append((session_index, val_sample, label, imp_id))

        return samples

def Val_collate_fn(batch):
    session_indices = []
    val_samples = []
    labels = []
    imp_ids = []

    for item in batch:
        if not item:
            continue

        for session_index, val_sample, label, imp_id in item:
            session_indices.append(session_index)
            val_samples.append(val_sample)
            labels.append(label) # 假设label是整数类型
            imp_ids.append(imp_id)

    # 将数据转化为PyTorch张量
    session_indices = torch.tensor(session_indices, dtype=torch.long)
    val_samples = torch.tensor(val_samples, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    imp_ids = torch.tensor(imp_ids,dtype=torch.float)

    return (session_indices, val_samples, labels, imp_ids)