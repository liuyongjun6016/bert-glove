import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def bert_embeddings(text_list,nid2index:dict):
    # 加载预训练的BERT模型和tokenizer
    save_path1 = '../inputdata/new_vocab/bert_title_abs_embedding.npy'
    save_path2 = '../inputdata/bert_word_embedding.npy'
    print("加载预训练的BERT模型和tokenizer...")
    model_dir = '../Bert'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)

    # 移动模型到GPU上（如果可用）
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # BytE CODING

    # 将 numpy 数组转换为列表
    text_list = text_list.tolist()#一个list，长度55239


    # # 对输入文本进行编码
    encoding = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=100)

    # 创建数据集
    dataset = TensorDataset(encoding['input_ids'], encoding['attention_mask'])

    # 设置更大的批量大小以提高GPU利用率
    batch_size = 20
    dataloader = DataLoader(dataset, batch_size=batch_size)

    sentences_embeddings = []
    count = 0

    embedding = np.array((len(nid2index),768),dtype=float)

    # 优化批次处理以减少GPU空闲时间
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch

            # 将批次移动到GPU上
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 获取模型输出
            outputs = model(input_ids, attention_mask=attention_mask)


            # 获取 [CLS] 标记的向量，作为句子的表征
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]
            # 768
            sentences_embeddings.append(sentence_embeddings.cpu().numpy())

            count+=1
            print('batch',count)

    # 将所有嵌入向量合并成一个数组
    sentences_embeddings_res = np.vstack(sentences_embeddings)

    list1 = np.random.normal(size=(768,))

    data_with_new_row = np.vstack([list1, sentences_embeddings_res])


    # 保存到文件
    print(data_with_new_row.shape)
    np.save(save_path1, data_with_new_row)
    print(f"句子嵌入向量已保存到 {save_path1}")

    # print(words_embeddings_res.shape)
    # np.save(save_path2, words_embeddings_res)
    # print(f"词嵌入向量已保存到 {save_path2}")

# data = np.load('../inputdata/bert_title_abs_embedding.npy')#(65238, 768)
#
# print(data.shape)

# list1 = np.random.normal(size=(768,))
#
# data_with_new_row = np.vstack([list1, data])
#
# np.save('../inputdata/bert_title_abs_embedding.npy', data_with_new_row)