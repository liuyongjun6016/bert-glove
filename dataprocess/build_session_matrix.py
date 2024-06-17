import ast
import os
import json
import pickle
import random
import argparse
import multiprocessing as mp
from typing import List, Dict, Set, Tuple

import tqdm
import torch
import numpy as np



def build_session_matrix(hist_dict, nid2index: Dict[str, int]):

    random.seed(7)

    count_session = 0
    session_dict = {}
    temp = 0
    for key,value in hist_dict.items():
        list1 = value.split('],')
        session_hitory_list = []
        for list_str in list1:
            if not list_str.endswith(']'):
                list_str += ']'
            py_list = ast.literal_eval(list_str)
            session_hitory_list.append(py_list)
        session_num = len(session_hitory_list)
        temp = temp + session_num
        session_dict[key] = list(range(count_session+1, temp+1))
        count_session = temp

    # print(count_session)#156073

    session2news = np.zeros((count_session+1, 50), dtype=int)

    # df是通过uid来提取一正四负样本
    for uid, value in hist_dict.items():
        if uid in session_dict:
            session_index_list = session_dict[uid]
            hitory_list = hist_dict[uid].split('],')

            session_hitory_list = []
            for list_str in hitory_list:
                if not list_str.endswith(']'):
                    list_str += ']'
                py_list = ast.literal_eval(list_str)
                session_hitory_list.append(py_list)
            for i in range(len(session_hitory_list)):
                session_index = session_index_list[i]
                session_list = session_hitory_list[i]
                indices = [nid2index.get(nid, 0) for nid in session_list]
                if len(indices) < 50:
                    indices += [0] * (50 - len(indices))
                else:
                    indices = indices[:50]
                session2news[session_index] = indices
        else:
            continue

    return session_dict, session2news

# def build_examples(rank: int,
#                    df: List[str],
#                    session_dict,
#                    nid2index: Dict[str, int]) -> None:
#
#     print("Loading sample data")
#     if rank == 0:
#         loader = tqdm.tqdm(df, desc="Building")
#     else:
#         loader = df
#
#     for row in loader:
#         row = json.loads(row)
#         uid = row["uid"]
#
#         if uid in session_dict:
#             session_index = session_dict[uid]
#         else:
#             continue
#
#         for pair in row["pairs"]:
#             news_id = pair[0]
#             pos_index = [nid2index[news_id]]
#             random_neg_ids = pair[1]
#             neg_index = [nid2index[nid] for nid in random_neg_ids]
#             sample_index = pos_index + neg_index
#
#
#         print(session_index)
#         print(sample_index)
#         print("********************")

        # data = Data(session_index,sample_index)




def main(args):
    train_f_hist = os.path.join("../inputdata/train-user_hist_dict.txt")#train-user_hist_dict.txt
    val_f_hist = os.path.join("../inputdata/val-user_hist_dict.txt")

    # Load user history dict
    # 通过hist_dict[UID]=[history]
    train_hist_dict = dict()
    lines = open(train_f_hist, "r", encoding="utf8").readlines()
    error_line = 0
    for l in lines:
        row = l.strip().split("\t")
        if len(row) == 1:
            error_line += 1
            continue
        train_hist_dict[row[0]] = row[1]
    print("train User history error Line: ", error_line)

    val_hist_dict = dict()
    lines2 = open(val_f_hist, "r", encoding="utf8").readlines()
    error_line2 = 0
    for l2 in lines2:
        row = l2.strip().split("\t")
        if len(row) == 1:
            error_line2 += 1
            continue
        val_hist_dict[row[0]] = row[1]
    print("val User history error Line: ", error_line2)



    with open("../inputdata/new_vocab/nid2index.bin", 'rb') as file:
        nid2index = pickle.load(file)

    session_dict,session2news = build_session_matrix(train_hist_dict, nid2index)
    np.save('../inputdata/session2news.npy', session2news)

    # 从文件加载 session_dict
    with open('../inputdata/session_dict.pkl', 'wb') as f:
        pickle.dump(session_dict, f)

    val_session_dict,val_session2news = build_session_matrix(val_hist_dict, nid2index)
    np.save('../inputdata/val_session2news.npy', val_session2news)

    # 从文件加载 session_dict
    with open('../inputdata/val_session_dict.pkl', 'wb') as f:
        pickle.dump(val_session_dict, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    # parser.add_argument("--max_hist_length", default=50, type=int, help="Max length of the click history of the user.")
    # parser.add_argument("--processes", default=10, type=int, help="Processes number")

    args = parser.parse_args()

    main(args)
