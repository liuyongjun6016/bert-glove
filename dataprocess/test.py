import os
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from database import ValDataset,Val_collate_fn

with open('../inputdata/new_vocab/nid2index.bin', 'rb') as file:
    nid2index = pickle.load(file)
val_samples = os.path.join("../MINDsmall/val/sessionBehaviors.tsv")
val_df = pd.read_csv(val_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
with open('../inputdata/val_session_dict.pkl', 'rb') as f:
    val_session_dict = pickle.load(f)
val_dataset = ValDataset(val_df, val_session_dict, nid2index)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=Val_collate_fn)


for batch in val_dataloader:
    session_indices, val_samples, labels = batch
    print("Session indices:", session_indices.shape)
    print("Validation samples:", val_samples.shape)
    print("Labels:", labels.shape)