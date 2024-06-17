import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.vocab import WordVocab

class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 200),
            nn.Tanh()
        )

        self.gate_layer = nn.Linear(200, 1)

    def forward(self, seqs, seq_masks=None):
        # [ batch_size, sequence_length, embedding_dim]
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)  # ??????squeeze(-1)????????
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        # [B, S]
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)  # ?????
        # [B, S, E] * [B, S, 1] = [B, S, E]
        h = seqs * p_attn
        # ??[B, E]
        output = torch.sum(h, dim=1)
        return output

def build_embedding_layer(pretrained_embedding_path, vocab, embedding_dim):
    num_embeddings = len(vocab)
    if pretrained_embedding_path != "":
        weights = np.load(pretrained_embedding_path)
        weights = torch.tensor(weights).float()
        assert list(weights.size()) == [num_embeddings, embedding_dim]
        print("load pre-trained embeddings.")
        return nn.Embedding.from_pretrained(weights, freeze=False)

    return nn.Embedding(num_embeddings, embedding_dim)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.word_hidden_size = 300
        self.news_bert_size = 768
        self.word_head_nums = 15
        self.neg_count = 4
        self.max_news_len = 80
        self.max_hist_len = 50
        self.max_session_len = 5


        # Init Layers
        self.session2news = torch.LongTensor(np.load('../inputdata/session2news.npy'))
        self.news2title = torch.LongTensor(np.load('../inputdata/news_title.npy'))  # './tempdata/news_title.npy'
        self.vocab = WordVocab.load_vocab('../inputdata/new_vocab/word_vocab.bin')
        self.val_session2news = torch.LongTensor(np.load('.0/inputdata/val_session2news.npy'))
        weights = np.load('../inputdata/bert_title_abs_embedding.npy')
        weights = torch.tensor(weights).float()
        self.title2bert_embedding = nn.Embedding.from_pretrained(weights, freeze=False)
        self.title2word_embedding = build_embedding_layer( pretrained_embedding_path='../inputdata/new_vocab/word_embeddings.bin.npy', vocab=self.vocab, embedding_dim=self.word_hidden_size)

        self.word_multi_head = nn.MultiheadAttention(self.word_hidden_size,self.word_head_nums,batch_first=True)
        self.titel_multi_head = nn.MultiheadAttention(self.word_hidden_size+self.news_bert_size,12,batch_first=True)
        self.session_multi_head = nn.MultiheadAttention(self.word_hidden_size+self.news_bert_size,12,batch_first=True)

        self.word_self_att = SelfAttend(self.word_hidden_size)
        self.title_self_att = SelfAttend(self.word_hidden_size+self.news_bert_size)
        self.session_self_att = SelfAttend(self.word_hidden_size+self.news_bert_size)





    def forward(self, session_indices, sample_indices):

        batch_size = session_indices.size()[0]

        news_seqs = self.session2news[session_indices]
        sample_bert_embedding = self.title2bert_embedding(news_seqs)#[batch, 5, 50, 768]
        title_seqs = self.news2title[news_seqs]
        sample_glove_title_embedding = self.title2word_embedding(title_seqs)#[batch, 5, 50, 80, 300]

        target_title_seqs = self.news2title[sample_indices]
        target_bert_embedding = self.title2bert_embedding(sample_indices)#[9, 5, 768]
        target_glove_title_embedding = self.title2word_embedding(target_title_seqs)#[batch, 5, 80, 300]

        sample_glove_title_embedding = sample_glove_title_embedding.reshape(-1, 80, 300)
        target_glove_title_embedding = target_glove_title_embedding.reshape(-1, 80, 300)

        sample_glove_title_embeddings,_ = self.word_multi_head(sample_glove_title_embedding,sample_glove_title_embedding,sample_glove_title_embedding)
        target_glove_title_embeddings,_ = self.word_multi_head(target_glove_title_embedding,target_glove_title_embedding,target_glove_title_embedding)

        sample_glove_embedding = self.word_self_att(sample_glove_title_embeddings)
        target_glove_embedding = self.word_self_att(target_glove_title_embeddings)

        sample_glove_embedding = sample_glove_embedding.reshape(-1, self.max_session_len, self.max_hist_len, 300)
        target_glove_embedding = target_glove_embedding.reshape(-1, self.neg_count+1, 300)

        sample_news_embedding = torch.cat([sample_bert_embedding, sample_glove_embedding], dim=-1)
        target_news_embedding = torch.cat([target_bert_embedding, target_glove_embedding], dim=-1)

        sample_news_embedding = sample_news_embedding.reshape(-1, self.max_hist_len, self.news_bert_size+self.word_hidden_size)
        sample_session_embedding, _ = self.titel_multi_head(sample_news_embedding,sample_news_embedding,sample_news_embedding)
        sample_session_embedding = self.title_self_att(sample_session_embedding)

        sample_session_embedding = sample_session_embedding.reshape(-1, self.max_session_len, self.news_bert_size+self.word_hidden_size)
        sample_user_embedding, _ = self.session_multi_head(sample_session_embedding,sample_session_embedding,sample_session_embedding)
        user_embedding = self.session_self_att(sample_user_embedding)

        user_hiddens = user_embedding.repeat(1, self.neg_count + 1).view(-1, self.word_hidden_size+self.news_bert_size)

        target_hiddens = target_news_embedding.view(-1, self.word_hidden_size+self.news_bert_size)

        logits = torch.sum(user_hiddens * target_hiddens, dim=-1)

        logits = logits.view(-1, self.neg_count + 1)

        return logits

    def training_step(self, data):
        # REQUIRED
        session_indices, sample_indices, labels = data

        logits = self.forward(session_indices, sample_indices)

        target = labels

        loss = F.cross_entropy(logits, target)

        return loss

    def predict(self, session_indices, val_sample_index):

        batch_size = session_indices.size()[0]

        news_seqs = self.val_session2news[session_indices]
        sample_bert_embedding = self.title2bert_embedding(news_seqs)#[batch, 5, 50, 768]
        title_seqs = self.news2title[news_seqs]
        sample_glove_title_embedding = self.title2word_embedding(title_seqs)#[batch, 5, 50, 80, 300]

        val_title_seqs = self.news2title[val_sample_index]
        val_bert_embedding = self.title2bert_embedding(val_sample_index)
        val_glove_title_embedding = self.title2word_embedding(val_title_seqs)

        sample_glove_title_embedding = sample_glove_title_embedding.reshape(-1, 80, 300)
        val_glove_title_embedding =val_glove_title_embedding.reshape(-1, 80, 300)

        sample_glove_title_embeddings, _ = self.word_multi_head(sample_glove_title_embedding,sample_glove_title_embedding,sample_glove_title_embedding)
        val_glove_title_embeddings, _ = self.word_multi_head(val_glove_title_embedding, val_glove_title_embedding, val_glove_title_embedding)

        sample_glove_embedding = self.word_self_att(sample_glove_title_embeddings)
        val_glove_embeddings = self.word_self_att(val_glove_title_embeddings)

        sample_glove_embedding = sample_glove_embedding.reshape(-1, self.max_session_len, self.max_hist_len, 300)
        val_glove_embedding =  val_glove_embeddings.reshape(-1, 1, 300)

        sample_news_embedding = torch.cat([sample_bert_embedding, sample_glove_embedding], dim=-1)
        val_news_embedding = torch.cat([val_bert_embedding, val_glove_embedding], dim=-1)

        sample_news_embedding = sample_news_embedding.reshape(-1, self.max_hist_len, self.news_bert_size+self.word_hidden_size)
        sample_session_embedding, _ = self.titel_multi_head(sample_news_embedding,sample_news_embedding,sample_news_embedding)
        sample_session_embedding = self.title_self_att(sample_session_embedding)

        sample_session_embedding = sample_session_embedding.reshape(-1, self.max_session_len, self.news_bert_size+self.word_hidden_size)
        sample_user_embedding, _ = self.session_multi_head(sample_session_embedding,sample_session_embedding,sample_session_embedding)
        user_embedding = self.session_self_att(sample_user_embedding)

        user_hiddens = user_embedding.view(-1, self.word_hidden_size+self.news_bert_size)

        target_hiddens = val_news_embedding.view(-1, self.word_hidden_size + self.news_bert_size)

        logits = torch.sum(user_hiddens * target_hiddens, dim=-1)

        return logits

    def validation_step(self, session_indices, val_sample_index):

        preds = self.predict(session_indices, val_sample_index)

        return preds

