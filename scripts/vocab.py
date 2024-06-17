import pickle
import tqdm
from collections import Counter


class TorchVocab(object):

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        self.itos = list(specials)#用来记录含有的词

        for tok in specials:#删除掉counter中的特殊符号
            del counter[tok]
        max_size = None if max_size is None else max_size + len(self.itos)#设置词汇表的大小

        # sort by frequency, then alphabetically 根据词频从高到低对词汇表进行排序
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos >>stoi是从高到低排序后的，每个词的索引表
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}#每个词一个索引

        self.vectors = None
        if vectors is not None:#如果提供预训练词向量，加载
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    #比较词汇表是否相等
    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    #返回词数量
    def __len__(self):
        return len(self.itos)

    #重新排序分配索引
    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    #将另一个词汇表扩展到本词汇表
    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)

    #将句子转化为序列的方法
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    #将序列转化回句子
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    # 声明静态方法不需实例化即可用
    @staticmethod
   #从文件中加载词汇表
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    #将词汇表保存在文件
    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1, lower=False):
        print("Building Vocab")
        self.lower = lower#false
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", " ").replace("\t", " ").split()
            for word in words:
                if self.lower:
                    word = word.lower()
                counter[word] += 1
        print("Total words: {}".format(len(counter)))
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    #将句子妆化为序列（即一系列词汇的索引）
    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        #如果是字符串，进行拆分
        if isinstance(sentence, str):
            sentence = sentence.split()

        #转化为小写
        if self.lower:
            #如果词汇不在词汇表中，则使用未知词汇的索引 unk_index
            seq = [self.stoi.get(word.lower(), self.unk_index) for word in sentence]
        else:
            seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        #保存结束符
        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq
        #记录元素长度
        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        #如果序列长度小于或等于指定的长度 seq_len，则将序列长度更新为最小值，并在序列末尾添加填充标记 <pad> 的索引，直到序列长度达到 seq_len
        elif len(seq) <= seq_len:
            origin_seq_len = min(len(seq), seq_len)
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        #如果序列长度大于指定的长度 seq_len，则将序列截断为指定长度
        else:
            origin_seq_len = min(len(seq), seq_len)
            seq = seq[:seq_len]
        #根据参数 with_len 决定是否返回序列的长度，如果需要返回长度，则以元组形式返回序列和原始序列长度，否则只返回序列
        return (seq, origin_seq_len) if with_len else seq

    #序列转化回句子
    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]
        return " ".join(words) if join else words

    #保存词汇表
    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()
    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)
    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)
