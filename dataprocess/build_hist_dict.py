import os
import argparse
from typing import List, Dict

import pandas as pd




def sample_user_history(hists: List[str], max_count: int = 10) -> List[str]:

    return hists[-max_count:]


def build_hist_dict(args: argparse.Namespace,
                    df: pd.DataFrame,
                    user_hist_dict: Dict[str, List[list]]) -> Dict[str, List[list]]:
    """
    Args:
        args: config
        df: user behavior data
        user_hist_dict: existed dict which map uid to history news list

    Returns:
        a new dict
    """
    for uid, hist in df[["uid", "hist"]].values:

        hist = str(hist).strip().split()
        if len(hist) == 0:
            continue

        # sampled_hist = sample_user_history(hist, args.max_hist_length)#取hist前50个
        sampled_hist = hist

        if uid in user_hist_dict:
            user_hist_dict[uid].append(sampled_hist)
        else:
            user_hist_dict[uid] = [sampled_hist]

    return user_hist_dict


def main(args):

    f = ["train" , "val"]

    for filename in f:

        behavior_path = os.path.join("../MINDsmall/", filename, "sessionBehaviors.tsv")
        out_path = os.path.join("../inputdata/", filename + "-" + args.fname)

        user_hist_dict = {}
        print("Building from {}".format(behavior_path))
        train_df = pd.read_csv(behavior_path, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
        train_df = train_df[train_df["hist"].isna() == False]  #去掉没有历史用户的记录
        user_hist_dict = build_hist_dict(args, train_df, user_hist_dict)  #用uid取history

        print("User Count: {}".format(len(user_hist_dict)))


        with open(out_path, "w", encoding="utf8") as fw:
            for uid, hist_list in user_hist_dict.items():
                # 将键和对应的历史记录列表转换为字符串，并写入文件中
                fw.write("{}\t{}\n".format(uid, ','.join(map(str, hist_list))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.

    parser.add_argument("--fname", default="user_hist_dict.txt", type=str, help="Output file name.")
    # parser.add_argument("--max_hist_length", default=50, type=int, help="Max length of the click history of the user.")

    args = parser.parse_args()

    main(args)
