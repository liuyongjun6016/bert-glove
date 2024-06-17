
import os
import json
import argparse
import pickle
import re

import pandas as pd
import numpy as np
from bert_embedding import bert_embeddings
from scripts.vocab import WordVocab


def word_tokenize(sent):
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def build_word_embeddings(vocab, pretrained_embedding, weights_output_file):
    # Load Glove embedding
    lines = open(pretrained_embedding, "r", encoding="utf8").readlines()
    emb_dict = dict() # 构建单词嵌入字典
    error_line = 0
    embed_size = 0
    for line in lines:
        row = line.strip().split()
        try:
            embedding = [float(w) for w in row[1:]]
            emb_dict[row[0]] = np.array(embedding)
            if embed_size == 0:
                embed_size = len(embedding)
        except:
            error_line += 1
    print("Error lines: {}".format(error_line))

    # embed_size = len(emb_dict.values()[0])
    # build embedding weights for model
    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for i, word in enumerate(vocab.itos):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(embed_size,))  # 未找到该单词则随机初始化
    print("Totally find {} words in pre-trained embeddings.".format(words_found))
    np.save(weights_output_file, weights_matrix)


def parse_ent_list(x):
    if str(x).strip() == "":
        return ''
    return ' '.join([k["WikidataId"] for k in json.loads(x)])



def main(cfg):
    # Build vocab
    print("Loading news info")
    f_train_news = os.path.join("../MINDsmall/train/news.tsv")
    f_dev_news = os.path.join("../MINDsmall/val/news.tsv")
    # f_test_news = os.path.join("../dataset/test/news.tsv")

    print("Loading training news")
    all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                           names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                           quoting=3)
    if os.path.exists(f_dev_news):
        print("Loading dev news")
        dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                               names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                               quoting=3)
        all_news = pd.concat([all_news, dev_news], ignore_index=True)

    # if os.path.exists(f_test_news):
    #     print("Loading testing news")
    #     test_news = pd.read_csv(f_test_news, sep="\t", encoding="latin1",
    #                             names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
    #                             quoting=3)
    #     all_news = pd.concat([all_news, test_news], ignore_index=True)

    all_news = all_news.drop_duplicates("newsid") # 删除重复项
    print("All news: {}".format(len(all_news)))

    df = all_news
    df = df.fillna(" ")
    # df['ents1'] = df['title_ents'].apply(lambda x: parse_ent_list(x))
    # df['ents2'] = df['abs_ents'].apply(lambda x: parse_ent_list(x))     # 提取实体信息并分别存储在'ents1'列和'ents2'列中
    # df["ent_list"] = df[[ "ents1", "ents2"]].apply(lambda x: " ".join(x), axis=1)       # 这四列按行合并为一个新列'ent_list'，并以空格分隔各个元素

    # 建立一个类，ent_vocab：
    # self.itos词汇表，list形式包含所有单词
    # self.stoi是对应itos，每个词的索引表
    # self.vectors 可以用来存储预训练的词向量

    # ent_vocab = WordVocab(df.ent_list.values, max_size=80000, min_freq=1, lower=cfg.lower)   # 80000，false
    # print("ENTITY VOCAB SIZE: {}".format(len(ent_vocab)))  # 检查entity频次
    # fpath = os.path.join("/", cfg.output, "entity_vocab.bin")
    # ent_vocab.save_vocab(fpath)

    # Building for text 将标题和摘要转化为单词列表
    df['title_token'] = df['title'].apply(lambda x: ' '.join(word_tokenize(x)))
    df['abs_token'] = df['abs'].apply(lambda x: ' '.join(word_tokenize(x)))
    df["text"] = df[["title_token", "abs_token"]].apply(lambda x: " ".join(x), axis=1)


    newsid = all_news["newsid"].unique()

    nid2index = {'<pad>': 0}

    for idx ,newsid in enumerate(newsid,start=1):
        nid2index[newsid] = idx

    file_path = '../inputdata/new_vocab/nid2index.bin'

    # 使用pickle将字典保存为二进制文件
    with open(file_path, 'wb') as file:
        pickle.dump(nid2index, file)

    #将title 和 abstract 拼接起来 为text

    # 建立一个类，word_vocab：
    # self.itos词汇表，list形式包含所有单词
    # self.stoi是对应itos，每个词的索引表
    # self.vectors 可以用来存储预训练的词向量


    # bert_embeddings(df.text.values,nid2index)


    # word_vocab = WordVocab(df.text.values, max_size=80000, min_freq=1, lower=cfg.lower)#80000，false
    # print("TEXT VOCAB SIZE: {}".format(len(word_vocab)))#检查word频次
    # f_text_vocab_path = os.path.join("../", cfg.output, "word_vocab.bin")
    # word_vocab.save_vocab(f_text_vocab_path)
    #
    #
    # # Build word embeddings
    # print("Building word embedding matrix")
    # pretrain_path = os.path.join("../", cfg.pretrain)
    # weight_path = os.path.join("../", cfg.output, "word_embeddings.bin")
    # build_word_embeddings(word_vocab, pretrain_path, weight_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path options.
    parser.add_argument("--pretrain", default="GloVE/glove.840B.300d.txt", type=str, help="Path of the raw review data file.")
    parser.add_argument("--output", default="inputdata/new_vocab/", type=str, help="Path of the training data file.")
    parser.add_argument("--lower", action='store_true')#是否是小写，使用是ture不使用是false

    args = parser.parse_args()

    main(args)
