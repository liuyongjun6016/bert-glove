import os
import json
import random
import pickle
import argparse
import re
from typing import Set, Dict
import numpy as np
import pandas as pd
import networkx as nx
from scripts.vocab import WordVocab


random.seed(7)

def parse_ent_list(x):
    if str(x).strip() == "":
        return ''
    return ' '.join([k["WikidataId"] for k in json.loads(x)])

def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []



def load_real_train_news(f_train_samples, f_train_hist):
    news_list_in_train = []
    lines = open(f_train_samples, "r", encoding="utf-8").readlines()
    for l in lines:
        j = json.loads(l)
        for pair in j["pairs"]:
            news_list_in_train.append(pair[0]) #positive
            news_list_in_train += pair[1] #nagetive
    
    lines = open(f_train_hist, "r").readlines()
    for l in lines:
        news_list_in_train += l.split()[1].split(',')
    news_list_in_train = set(news_list_in_train)
    return news_list_in_train

ROOT_PATH = "../"

def main(args):

    f_train_news = os.path.join(ROOT_PATH, "MINDsmall/train/news.tsv")
    f_dev_news = os.path.join(ROOT_PATH, "MINDsmall/val/news.tsv")
    # f_test_news = os.path.join(ROOT_PATH, "dataset/test/news.tsv")
    f_title_matrix = os.path.join(ROOT_PATH, "inputdata/news_title.npy")


    f_out = os.path.join(ROOT_PATH, "inputdata/news_dict-{}.txt".format(args.fvocab))

    print("Loading training news")
    train_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
    all_news = train_news.copy(deep=True)
    dev_news = None
    if os.path.exists(f_dev_news):
        print("Loading dev news")
        dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
        all_news = pd.concat([all_news, dev_news], ignore_index=True)

    # test_news = None
    # if os.path.exists(f_test_news):
    #     print("Loading testing news")
    #     test_news = pd.read_csv(f_test_news, sep="\t", encoding="utf-8",
    #                             names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
    #                             quoting=3)
    #     all_news = pd.concat([all_news, test_news], ignore_index=True)
    # all_news = all_news.drop_duplicates("newsid")
    # print("All news: {}".format(len(all_news)))

    # Build news_id => neighbor entity set
    # Load entity vocab and graph


    # Build news_id => title, news_id => cates
    newsid_title_abs_dict = {}

    for row in all_news[["newsid", "title", "abs"]].values:
        title = " ".join(word_tokenize(row[1])[:30])
        abs = " ".join(word_tokenize(row[2])[:50])
        newsid_title_abs_dict[row[0]] = title+abs


    # Title matrix: news index => title word indices #将新闻标题变成index
    f_word_vocab = os.path.join(ROOT_PATH, "inputdata/new_vocab/word_vocab.bin")
    word_vocab = WordVocab.load_vocab(f_word_vocab)

    with open("../inputdata/new_vocab/nid2index.bin", 'rb') as file:
        nid2index = pickle.load(file)


    # with open(f_out, "w", encoding="utf8") as fw:
    #     default_info = {
    #         "glove_title_abs": ['<pad>'],
    #         "bert_title_abs": [0]
    #     }
    #     fw.write("PAD\t{}\n".format(json.dumps(default_info, ensure_ascii=False)))
    #
    #     for news_id in newsid_title_abs_dict:
    #         cur_title = newsid_title_abs_dict.get(news_id, "")
    #         if isinstance(cur_title, str):
    #             cur_title = cur_title.split()
    #         title = [word_vocab.stoi.get(word.lower(), word_vocab.unk_index) for word in cur_title]
    #
    #
    #         news_info = {
    #             "title": title,
    #             "abs": abs,
    #             "ents": entity,
    #         }
    #         fw.write("{}\t{}\n".format(news_id, json.dumps(news_info, ensure_ascii=False)))



    news2title = np.zeros((len(newsid_title_abs_dict) + 1, 30+50), dtype=int)#（len(newsid_title_dict) + 1）*20
    news2nid = {}
    news2nid['PAD'] = 0
    news2title[0], cur_len = word_vocab.to_seq('<pad>', seq_len=80, with_len=True)
    news_index = 1

    with open(f_out, "w", encoding="utf8") as fw:
        default_info = {
            "glove_title_abs": [0, 1],
            "bert_title_abs": [0]
        }
        fw.write("PAD\t{}\n".format(json.dumps(default_info, ensure_ascii=False)))

        for news_id in newsid_title_abs_dict:
            news2nid[news_id] = news_index
            cur_title = newsid_title_abs_dict.get(news_id, "")
            news2title[news_index], cur_len = word_vocab.to_seq(cur_title, seq_len=80, with_len=True)

            news_info = {
                    "glove_title_abs": [news2nid[news_id], cur_len],
                    "bert_title_abs": [nid2index[news_id]]
                }
            fw.write("{}\t{}\n".format(news_id, json.dumps(news_info, ensure_ascii=False)))

            news_index += 1

    np.save(f_title_matrix, news2title)
    print("title embedding: ", news2title.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.

    parser.add_argument("--fvocab", default="new_vocab", type=str, help="Path of the output dir.")
    # parser.add_argument("--max_title_len", default=30, type=int, help="Max length of the title.")

    args = parser.parse_args()

    main(args)
