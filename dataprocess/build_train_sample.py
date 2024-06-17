import os
import json
import random
import argparse

import tqdm
import pandas as pd

random.seed(7)


def main(args):
    f_behaviors = os.path.join("../MINDsmall/train/sessionBehaviors.tsv")
    f_out = os.path.join("../inputdata/train_samples.tsv")

    df = pd.read_csv(f_behaviors, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    with open(f_out, "w", encoding="utf-8") as fw:
        for row in tqdm.tqdm(df[["uid", "imp"]].values, desc="Building"):
            uid = row[0]
            samples = row[1].strip().split()
            pos_news_ids, neg_news_ids = list(), list()
            for sample in samples:
                news_id, label = sample.split("-")[:2]
                if label == "1":
                    pos_news_ids.append(news_id)
                else:
                    neg_news_ids.append(news_id)

            if len(neg_news_ids) < args.neg_num:
                continue

            train_pairs = []
            for pos_news_id in pos_news_ids[:1]:
                random_neg_ids = random.sample(neg_news_ids, args.neg_num)
                train_pairs.append((pos_news_id, random_neg_ids))

            j = {
                "uid": uid,
                "pairs": train_pairs
            }
            fw.write(json.dumps(j, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--neg_num", default=4, type=int, help="Max neg samples according to one pos sample.")
    args = parser.parse_args()

    main(args)