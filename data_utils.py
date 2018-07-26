import numpy as np
import pickle as pkl
import codecs, json, os, sys, jieba
from collections import OrderedDict, Counter

class jieba_api(object):
    def __init__(self):
        print("----------using jieba cut tool---------")

    def init_config(self, config):
        self.config = config

    def build_tool(self):
        dict_path = self.config.get("user_dict", None)
        if dict_path is not None:
            jieba.load_userdict(dict_path)

    def cut(self, text):
        return " ".join(list(jieba.cut(text.strip())))

class cut_tool_api(object):
    def __init__(self):
        print("----------using naive cut tool---------")

    def init_config(self, config):
        self.config = config

    def build_tool(self):
        pass

    def cut(self, text):
        out = []
        for word in text.strip():
            if len(word) >= 1:
                out.append(word)
        return " ".join(out)

class cut_api(object):
    def __init__(self, config):
        self.config = config
    def cut(self, text):
        return "".join(text.split("\n"))

def initialize_embedding(dictionary_path, char_dim):
    with codecs.open(dictionary_path, "r", "utf-8") as frobj:
        words = frobj.read().splitlines()
        id2token = OrderedDict()
        token2id = OrderedDict()
        token2id["<PAD>"] = 0
        id2token[0] = "<PAD>"
        token2id["<UNK>"] = 1
        id2token[1] = "<UNK>"
        for idx, word in enumerate(words):
            token2id[word] = idx + 2
            id2token[idx+2] = word
        vocab_size = len(list(token2id.keys()))
        
        embedding = np.random.uniform(low=-0.01, high=0.01, size=(vocab_size, char_dim))
    return embedding

def read_emebdding(embedding_path, dictionary=None):
    if sys.version_info < (3, ):
        w2v = pkl.load(open(embedding_path, "rb"))
    else:
        w2v = pkl.load(open(embedding_path, "rb"), encoding="iso-8859-1")
    if dictionary:
        with codecs.open(dictionary, "r", "utf-8") as frobj:
            words = frobj.read().splitlines()
            id2token = OrderedDict()
            token2id = OrderedDict()
            token2id["<PAD>"] = 0
            id2token[0] = "<PAD>"
            token2id["<UNK>"] = 1
            id2token[1] = "<UNK>"
            for idx, word in enumerate(words):
                token2id[word] = idx + 2
                id2token[idx+2] = word
    else:
        id2token = OrderedDict()
        token2id = OrderedDict()
        token2id["<PAD>"] = 0
        id2token[0] = "<PAD>"
        token2id["<UNK>"] = 1
        id2token[1] = "<UNK>"
        for idx, word in enumerate(list(w2v.keys())):
            token2id[word] = idx + 2
            id2token[idx+2] = word
            
    vocab_size = len(list(token2id.keys()))
    embed_dim = w2v[list(w2v.keys())[0]].shape[0]
    embedding = np.random.normal(low=-0.1, high=0.1, size=(vocab_size, embed_dim))
    for idx, token in enumerate(list(token2id.keys())):
        if token in w2v:
            embedding[idx] = w2v[token]
    return [w2v, token2id, id2token, embedding]

def read_data(data_path, mode, word_cut_api, data_cleaner_api):
    with codecs.open(data_path, "r", "utf-8") as frobj:
        lines = frobj.read().splitlines()
        corpus_anchor = []
        corpus_check = []
        gold_label = []
        anchor_len = []
        check_len = []
        for line in lines:
            content = line.split()
            if mode == "train" or mode == "test":
                if len(content) >= 3:
                    try:
                        sent1 = content[0]
                        sent2 = content[1] 
                        label = int(content[2])
                        if label == 1 or label == 0:
                            sent1 = data_cleaner_api.clean(sent1)
                            sent2 = data_cleaner_api.clean(sent2)
                            corpus_anchor.append(" ".join(list(word_cut_api.cut(sent1))))
                            corpus_check.append(" ".join(list(word_cut_api.cut(sent2))))
                            gold_label.append(label)
                            anchor_len.append(len(sent1))
                            check_len.append(len(sent2))
                        else:
                            continue
                    except:
                        continue
            else:
                if len(content) >= 2:
                    sent1 = content[0]
                    sent2 = content[1] 
                    sent1 = data_cleaner_api.clean(sent1)
                    sent2 = data_cleaner_api.clean(sent2)
                    corpus_anchor.append(" ".join(list(word_cut_api.cut(sent1))))
                    corpus_check.append(" ".join(list(word_cut_api.cut(sent2))))
                    anchor_len.append(len(sent1))
                    check_len.append(len(sent2))
        return [corpus_anchor, corpus_check, gold_label, anchor_len, check_len]
    
def utt2id(utt, token2id, max_length):
    utt2id_list = [token2id["<PAD>"]] * max_length
    for index, word in enumerate(utt.split()):
        if word in token2id:
            utt2id_list[index] = token2id[word]
        else:
            utt2id_list[index] = token2id["<UNK>"]
    return utt2id_list

def id2utt(uttid_list, id2token):
    utt = u""
    for index, idx in enumerate(uttid_list):
        if idx == 0:
            break
        else:
            utt += id2token[idx]
    return utt

def corpus_to_vocab(corpus_data, embed_size, vocab_size=None):
    if vocab_size:
        vocab_size = vocab_size
    else:
        vocab_size = 100000
    corpus_anchor, corpus_check = corpus_data
    total_corpus = []
    for item in corpus_anchor:
        total_corpus.extend(item.split())
    for item in corpus_check:
        total_corpus.extend(item.split())
    vocab_counter = Counter(total_corpus)
    most_common_words = vocab_counter.most_common(vocab_size)
    id2token = OrderedDict()
    token2id = OrderedDict()
    token2id["<PAD>"] = 0
    id2token[0] = "<PAD>"
    token2id["<UNK>"] = 1
    id2token[1] = "<UNK>"
    for index, item in enumerate(most_common_words):
        token2id[item[0]] = index + 2
        id2token[index+2] = item[0]
        
    vocab_size = len(list(token2id.keys()))
    embedding = np.random.uniform(low=-0.1, high=0.1, size=(vocab_size, embed_size))
    return [token2id, id2token, embedding, vocab_counter]

def w2v_to_corpus_vocab(w2v, corpus_data, mode="corpus", vocab_size=None):
    if mode == "corpus":
        [token2id, id2token, 
        embedding, vocab_counter] = corpus_to_vocab(corpus_data,
                                   w2v[list(w2v.keys())[0]].shape[0], 
                                  vocab_size=vocab_size)
    elif mode == "w2v":
        print("-----------using w2v not corpus----------")
        vocab_counter = None
        token2id = OrderedDict()
        id2token = OrderedDict()
        token2id["<PAD>"] = 0
        id2token[0] = "<PAD>"
        token2id["<UNK>"] = 1
        id2token[1] = "<UNK>"

        for idx, token in enumerate(list(w2v.keys())):
            token2id[token] = idx + 2
            id2token[idx+2] = token
        vocab_size = len(list(token2id.keys()))
        embed_size = w2v[list(w2v.keys())[0]].shape[0]
        embedding = np.random.uniform(low=-0.1, 
            high=0.1, size=(vocab_size, embed_size))
    cnt = 0
    for word in token2id:
        if word in w2v:
            embedding[token2id[word]] = w2v[word]
            cnt += 1
    print("------------total involving w2v------------", cnt)
    return [token2id, id2token, embedding, vocab_counter]
    
    
    
    

