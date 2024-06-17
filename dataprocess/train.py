import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from database import CustomDataset, collate_fn, ValDataset, Val_collate_fn
from torch.utils.data import Dataset, DataLoader
from model import Model
from train_and_val_util.eval_util import group_labels, cal_metric

# model
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# train
f_samples = os.path.join("../inputdata/train_samples.tsv")  # train_sample.tsv
df = open(f_samples, "r", encoding="utf-8").readlines()
with open('../inputdata/session_dict.pkl', 'rb') as f:
    session_dict = pickle.load(f)
with open('../inputdata/new_vocab/nid2index.bin', 'rb') as file:
    nid2index = pickle.load(file)
train_dataset = CustomDataset(df, session_dict, nid2index)
dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)

# val
val_samples = os.path.join("../MINDsmall/val/sessionBehaviors.tsv")
val_df = pd.read_csv(val_samples, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
with open('../inputdata/val_session_dict.pkl', 'rb') as f:
    val_session_dict = pickle.load(f)
val_dataset = ValDataset(val_df, val_session_dict, nid2index)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=Val_collate_fn)


val_count = 5000
step_count = 0
val_nums = 1000

for epoch in range(15):

    enum_dataloader = enumerate(tqdm(dataloader, total=len(dataloader), desc="EP-{} train".format(epoch)))
    for i, data in enum_dataloader:
        model.train()
        model.zero_grad()
        loss = model.training_step(data)
        loss.backward()
        print(loss.item())
        optimizer.step()
        model.zero_grad()
        step_count += 1

        if step_count % val_count == 0:
            val_dataloader = enumerate(tqdm(val_dataloader, total=len(val_dataloader), desc="EP-{} val".format(epoch)))
            model.eval()
            with torch.no_grad():
                preds, truths, imp_ids = list(), list(), list()
                for j, data in val_dataloader:
                    if j % val_nums == 0:
                        break
                    session_indices, val_sample_index, label, imp_ids = data
                    pred = model.validation_step(session_indices, val_sample_index)
                    preds += pred.cpu().numpy().tolist()
                    truths += label.long().cpu().numpy().tolist()

                all_labels, all_preds = group_labels(truths, preds, imp_ids)
                metric_list = [x.strip() for x in 'group_auc || mean_mrr || ndcg@5;10'.split("||")]
                ret = cal_metric(all_labels, all_preds, metric_list)
                with open('../checkpoint/session-base.txt', 'a') as f:
                    for metric, val in ret.items():
                        print("Epoch: {}, {}: {}".format(epoch, metric, val))
                        f.write("Epoch: {}, {}: {}\n".format(epoch, metric, val))
                    f.write("------------------\n")
