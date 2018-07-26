# -*- coding: utf-8 -*-
from data_utils import (read_emebdding, read_data, 
                    utt2id, corpus_to_vocab, 
                    w2v_to_corpus_vocab, id2utt, initialize_embedding)
import jieba, codecs
import numpy as np
from data_cleaner import DataCleaner
import pickle as pkl
import tensorflow as tf

from bimpm import BIMPM
from match_pyramid import MatchPyramid
from siamese_cnn import SiameseCNN
from siamese_lstm import SiameseLSTM
from bimpm_new import BIMPM_NEW
from disan import DiSAN
from biblosa import BiBLOSA
from dan import DAN
from qa_cnn import QACNN
from dan_fast import DANFast
from transformer_encoder import TransformerEncoder
from lstm_match_pyramid import LSTMMatchPyramid

from model_api import ModelAPI

import os

from data_utils import cut_api

embed_size = 256

# train_data_path = "/data/xuht/duplicate_sentence/chinese_duplicate/train.txt"
train_data_path = "/data/xuht/guoxin/poc/data/train.txt"

w2v_path = "/data/xuht/word2vec_model/word2vec_from_weixin/word2vec/wx_vector_char.pkl"

#w2v_path = "/data/xuht/word2vec/word2vec/w2v_64.pkl"

max_length = 200

data_clearner_api = DataCleaner({})

#jieba.load_userdict("/data/xuht/word2vec_model/word2vec_from_weixin/word2vec/wx_vector_dict.txt")

cut_tool = cut_api("")


[train_anchor, train_check, train_label, train_anchor_len, train_check_len] = read_data(train_data_path, "train", cut_tool, data_clearner_api)

w2v = pkl.load(codecs.open(w2v_path, "rb"))

[token2id, id2token, embedding, vocab_counter] = w2v_to_corpus_vocab(w2v, [train_anchor, train_check], mode="w2v", vocab_size=20000)
#embedding = initialize_embedding("/data/xuht/word2vec_model/word2vec_from_weixin/word2vec/wx_vector_dict_char.txt", 10000)

print("--------embedding size------", embedding.shape)

vocab_size = embedding.shape[0]


train_utt2ids_anchor = [utt2id(item, token2id, max_length) for item in train_anchor]
train_utt2ids_check = [utt2id(item, token2id, max_length) for item in train_check]

train_anchor_matrix = np.asarray(train_utt2ids_anchor).astype(np.int32)
train_check_matrix = np.asarray(train_utt2ids_check).astype(np.int32)
train_label_matrix = np.asarray(train_label).astype(np.int32)
train_anchor_len_matrix = np.asarray(train_anchor_len).astype(np.int32)
train_check_len_matrix = np.asarray(train_check_len).astype(np.int32)

# dev_data_path = "/data/xuht/duplicate_sentence/chinese_duplicate/dev.txt"
dev_data_path = "/data/xuht/guoxin/poc/data/dev.txt"
[dev_anchor, dev_check, dev_label, dev_anchor_len, dev_check_len] = read_data(dev_data_path, "train", cut_tool, data_clearner_api)
dev_utt2ids_anchor = [utt2id(item, token2id, max_length) for item in dev_anchor]
dev_utt2ids_check = [utt2id(item, token2id, max_length) for item in dev_check]
dev_anchor_matrix = np.asarray(dev_utt2ids_anchor).astype(np.int32)
dev_check_matrix = np.asarray(dev_utt2ids_check).astype(np.int32)
dev_label_matrix = np.asarray(dev_label).astype(np.int32)
dev_anchor_len_matrix = np.asarray(dev_anchor_len).astype(np.int32)
dev_check_len_matrix = np.asarray(dev_check_len).astype(np.int32)

print("--------------begin to train-----------------")

model_type = "lstm_match_pyramid"
if model_type == "bimpm":

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"bimpm",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"BIMPM",
            "num_layers":1,
            "optimizer":"adam",
            "learning_rate":1e-3,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "loss_type":"cross_entropy",
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":5,
            "weight_decay":5e-5,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm1/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm1/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm1", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm1")

    api.load_config()
    model = BIMPM()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "siamese_cnn":
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"siamese_cnn",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"SiameseCNN",
            "num_layers":2,
            "optimizer":"rmsprop",
            "learning_rate":1e-3,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "loss_type":"cross_entropy",
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":5,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_cnn1/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_cnn1/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_cnn1", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_cnn1")

    api.load_config()
    model = SiameseCNN()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                    train_anchor_len_matrix, train_check_len_matrix],
                    [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                    dev_anchor_len_matrix, dev_check_len_matrix])

    t = dev_anchor_matrix[0:10]
    s = dev_check_matrix[0:10]
    label = dev_label_matrix[0:10]

    s1, s2 = api.infer_step([train_anchor_matrix[0:10], train_check_matrix[0:10], None, 
                       train_anchor_len_matrix[0:10], train_check_len_matrix[0:10]])

    print(s2, label)

    api.load_model("latest")
    s1, s2 = api.infer_step([train_anchor_matrix[0:10], train_check_matrix[0:10], None, 
                       train_anchor_len_matrix[0:10], train_check_len_matrix[0:10]])

    print(s2, label)
    features = api.model.infer_features(api.sess, [train_anchor_matrix[0:10], train_check_matrix[0:10], None, 
                       train_anchor_len_matrix[0:10], train_check_len_matrix[0:10]],
                       1.0, "infer")

    print(features)


elif model_type == "siamese_lstm_l2":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"siamese_lstm_l2",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"SiameseLSTM_l2",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-4,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "ema":False,
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "loss_type":"contrastive_loss",
            "distance_metric":"l2_similarity",
            "feature_type":"last_pool",
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric")

    api.load_config()
    model = SiameseLSTM()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])
    
elif model_type == "siamese_lstm_l1":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"siamese_lstm_l1",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"SiameseLSTM_l1",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-3,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "loss_type":"contrastive_loss",
            "distance_metric":"l1_similarity",
            "feature_type":"last_pool",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":600,
            "gpu_ratio":0.9}

    print("--------------siamese lstm l1 contrastive metric----------------------")

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l1_contrastive_metric/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l1_contrastive_metric/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l1_contrastive_metric", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/siamese_lstm_l1_contrastive_metric")

    api.load_config()
    model = SiameseLSTM()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

    t = dev_anchor_matrix[0:10]
    s = dev_check_matrix[0:10]
    label = dev_label_matrix[0:10]

    s1, s2 = api.infer_step([train_anchor_matrix[0:10], train_check_matrix[0:10], None, 
                       train_anchor_len_matrix[0:10], train_check_len_matrix[0:10]])

    print(s2, label)

    api.load_model("latest")
    s1, s2 = api.infer_step([train_anchor_matrix[0:10], train_check_matrix[0:10], None, 
                       train_anchor_len_matrix[0:10], train_check_len_matrix[0:10]])

    print(s2, label)

elif model_type == "match_pyramid":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"match_pyramid",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "num_layers":2,
            "model_type":"MatchPyramid",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-4,
            "var_decay":0.999,
            "decay":0.9,
            "weight_decay":5e-5,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "loss_type":"cross_entropy",
            "distance_metric":"l1_similarity",
            "feature_type":"last_pool",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/match_pyramid/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/match_pyramid/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/match_pyramid", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/match_pyramid")

    api.load_config()
    model = MatchPyramid()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])


elif model_type == "disan":
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"disan",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"DiSAN",
            "num_layers":2,
            "optimizer":"adadelta",
            "learning_rate":0.5,
            "dropout_keep_prob":0.8,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":5,
            "loss_type":"cross_entropy",
            "weight_decay":5e-5,
            "out_channel_dims":[10,10,10],
            "filter_heights":[1,3,5],
            "hidden_units":100,
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":2000,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/disan/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/disan/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/disan", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/disan")

    api.load_config()
    model = DiSAN()
    api.build_graph(model, device="/gpu:0")

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])


elif model_type == "bimpm_new":

    OPTIONS = {
            "batch_size": 60,
            "max_epochs": 100,
            "learning_rate": 0.001,
            "optimize_type": "adam",
            "lambda_l2": 0.0,
            "grad_clipper": 10.0,

            "context_layer_num": 1,
            "context_lstm_dim": 100,
            "aggregation_layer_num": 1,
            "aggregation_lstm_dim": 100,

            "with_full_match": True,
            "with_maxpool_match": True,
            "with_max_attentive_match": True,
            "with_attentive_match": True,

            "with_cosine": True,
            "with_mp_cosine": True,
            "cosine_MP_dim": 5,

            "att_dim": 50,
            "att_type": "symmetric",

            "highway_layer_num": 1,
            "with_highway": True,
            "with_match_highway": True,
            "with_aggregation_highway": True,

            "use_cudnn": True,

            "with_moving_average": False}

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "options":OPTIONS,
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"bimpm",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "ema":False,
            "l2_reg":1e-5,
            "model_type":"BIMPM_NEW",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-3,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":20,
            "loss_type":"cross_entropy",
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":5,
            "weight_decay":5e-5,
            "validation_step":600,
            "gpu_ratio":0.9,
            "grad_clipper":10.0,
            "weight_decay":5e-5}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm_new1/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm_new1/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm_new1", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/bimpm_new1")

    api.load_config()
    model = BIMPM_NEW()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "biblosa":
    print("----training-----", model_type)
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"biblosa",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"BiBLOSA",
            "num_layers":2,
            "optimizer":"adadelta",
            "learning_rate":0.5,
            "dropout_keep_prob":0.8,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "ema":False,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":5,
            "loss_type":"cross_entropy",
            "context_fusion_method":"block",
            "block_len":None,
            "weight_decay":5e-5,
            "out_channel_dims":[50,50,50],
            "filter_heights":[1,3,5],
            "hidden_units":100,
            "persp_method":"pooling",
            "persp_num":4,
            "method_index":1,
            "use_bi":True,
            "batch_norm":False,
            "activation":"relu",
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":2000,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa")

    api.load_config()
    model = BiBLOSA()
    api.build_graph(model, device="/gpu:0")

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "biblosa_disa":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"biblosa_disa",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"BiBLOSA",
            "num_layers":2,
            "optimizer":"adadelta",
            "learning_rate":0.5,
            "dropout_keep_prob":0.8,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "ema":False,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":5,
            "loss_type":"cross_entropy",
            "context_fusion_method":"disa",
            "block_len":None,
            "weight_decay":5e-5,
            "out_channel_dims":[50,50,50],
            "filter_heights":[1,3,5],
            "hidden_units":100,
            "persp_method":"pooling",
            "persp_num":4,
            "method_index":1,
            "use_bi":True,
            "batch_norm":False,
            "activation":"relu",
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":2000,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa_disa/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa_disa/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa_disa", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/biblosa_disa")

    api.load_config()
    model = BiBLOSA()
    api.build_graph(model, device="/gpu:0")

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "dan":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
        "vocab_size":vocab_size,
        "max_length":200,
        "emb_size":embed_size,
        "extra_symbol":["<PAD>", "<UNK>"],
        "scope":"dan",
        "num_features":100,
        "num_classes":2,
        "filter_width":4,
        "di":50,
        "l2_reg":1e-5,
        "model_type":"DAN",
        "num_layers":2,
        "optimizer":"adam",
        "learning_rate":1e-4,
        "var_decay":0.999,
        "decay":0.9,
        "mode":"train",
        "sent1_psize":3,
        "sent2_psize":10,
        "feature_dim":20,
        "batch_size":30,
        "aggregation_rnn_hidden_size":100,
        "hidden_units":100,
        "rnn_cell":"lstm_cell",
        "dropout_keep_prob":0.8,
        "max_epoch":100,
        "early_stop_step":5,
        "weight_decay":5e-5,
        "validation_step":600,
        "hidden_units":100,
        "rnn_cell":"lstm_cell",
        "dropout_keep_prob":0.8,
        "loss_type":"contrastive_loss",
        "distance_metric":"l2_similarity",
        "feature_type":"last_pool",
        "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan")

    api.load_config()
    model = DAN()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "transformer_encoder":
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    config = {
        "vocab_size":vocab_size,
        "max_length":200,
        "emb_size":embed_size,
        "extra_symbol":["<PAD>", "<UNK>"],
        "scope":"transformer_encoder",
        "num_features":100,
        "num_classes":2,
        "filter_width":4,
        "di":50,
        "l2_reg":1e-5,
        "ema":False,
        "model_type":"TransformerEncoder",
        "num_layers":2,
        "optimizer":"adam",
        "num_blocks":6,
        "num_heads":8,
        "learning_rate":1e-4,
        "var_decay":0.999,
        "decay":0.9,
        "mode":"train",
        "sent1_psize":3,
        "sent2_psize":10,
        "feature_dim":20,
        "batch_size":30,
        "aggregation_rnn_hidden_size":100,
        "hidden_units":embed_size,
        "rnn_cell":"lstm_cell",
        "max_epoch":100,
        "early_stop_step":5,
        "weight_decay":5e-5,
        "validation_step":600,
        "sent_enc":"summarization",
        "dropout_keep_prob":0.8,
        "loss_type":"contrastive_loss",
        "distance_metric":"l2_similarity",
        "feature_type":"last_pool",
        "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/transformer_encoder/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/transformer_encoder/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/transformer_encoder", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/transformer_encoder")

    api.load_config()
    model = TransformerEncoder()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

if model_type == "qa_cnn":

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":max_length,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"qa_cnn",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "model_type":"QACNN",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-4,
            "var_decay":0.999,
            "decay":0.9,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "loss_type":"contrastive_loss",
            "distance_metric":"l2_similarity",
            "feature_type":"last_pool",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/qacnn/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/qacnn/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/qacnn", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/qacnn")

    api.load_config()
    model = QACNN()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])

elif model_type == "dan_fast":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = {
        "vocab_size":vocab_size,
        "max_length":200,
        "emb_size":embed_size,
        "extra_symbol":["<PAD>", "<UNK>"],
        "scope":"dan_fast",
        "num_features":100,
        "num_classes":2,
        "filter_width":4,
        "di":50,
        "l2_reg":1e-5,
        "model_type":"DANFast",
        "num_layers":2,
        "optimizer":"adam",
        "learning_rate":1e-4,
        "var_decay":0.999,
        "decay":0.9,
        "mode":"train",
        "sent1_psize":3,
        "sent2_psize":10,
        "feature_dim":20,
        "batch_size":30,
        "aggregation_rnn_hidden_size":100,
        "hidden_units":100,
        "rnn_cell":"lstm_cell",
        "dropout_keep_prob":0.8,
        "max_epoch":100,
        "early_stop_step":5,
        "weight_decay":5e-5,
        "validation_step":600,
        "hidden_units":100,
        "rnn_cell":"lstm_cell",
        "dropout_keep_prob":0.8,
        "loss_type":"contrastive_loss",
        "distance_metric":"l2_similarity",
        "feature_type":"last_pool",
        "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan_fast/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan_fast/config.json", "w"))


    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan_fast", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/dan_fast")

    api.load_config()
    model = DANFast()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])


elif model_type == "lstm_match_pyramid":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    config = {
            "vocab_size":vocab_size,
            "max_length":200,
            "emb_size":embed_size,
            "extra_symbol":["<PAD>", "<UNK>"],
            "scope":"lstm_match_pyramid",
            "num_features":100,
            "num_classes":2,
            "filter_width":4,
            "di":50,
            "l2_reg":1e-5,
            "num_layers":2,
            "model_type":"LSTMMatchPyramid",
            "num_layers":2,
            "optimizer":"adam",
            "learning_rate":1e-4,
            "var_decay":0.999,
            "decay":0.9,
            "weight_decay":5e-5,
            "mode":"train",
            "sent1_psize":3,
            "sent2_psize":10,
            "feature_dim":20,
            "batch_size":30,
            "aggregation_rnn_hidden_size":100,
            "hidden_units":100,
            "rnn_cell":"lstm_cell",
            "loss_type":"cross_entropy",
            "distance_metric":"l1_similarity",
            "feature_type":"last_pool",
            "rnn_cell":"lstm_cell",
            "dropout_keep_prob":0.8,
            "max_epoch":100,
            "early_stop_step":10,
            "validation_step":600,
            "gpu_ratio":0.9}

    pkl.dump({"token2id":token2id,
       "id2token":id2token,
       "embedding_matrix":embedding}, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/lstm_match_pyramid/emb_mat.pkl", "wb"))

    import json
    json.dump(config, open("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/lstm_match_pyramid/config.json", "w"))

    api = ModelAPI("/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/lstm_match_pyramid", 
              "/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models/lstm_match_pyramid")

    api.load_config()
    model = LSTMMatchPyramid()
    api.build_graph(model)

    api.train_step([train_anchor_matrix, train_check_matrix, train_label_matrix, 
                   train_anchor_len_matrix, train_check_len_matrix],
                   [dev_anchor_matrix, dev_check_matrix, dev_label_matrix, 
                   dev_anchor_len_matrix, dev_check_len_matrix])