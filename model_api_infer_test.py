# -*- coding: utf-8 -*-
from data_utils import (read_emebdding, read_data, utt2id, corpus_to_vocab, w2v_to_corpus_vocab, id2utt)
import jieba, codecs, os
import numpy as np
from data_cleaner import DataCleaner
import pickle as pkl
import tensorflow as tf

from bimpm import BIMPM
from match_pyramid import MatchPyramid
from siamese_cnn import SiameseCNN
from siamese_lstm import SiameseLSTM

from model_api import ModelAPI
from data_utils import cut_api

os.environ["CUDA_VISIBLE_DEVICES"]="0"
cut_tool = cut_api("")

# api = ModelAPI("/data/xuht/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric", 
#               "/data/xuht/duplicate_sentence_model/duplicate_models/siamese_lstm_l2_contrastive_metric")

# api.load_config()
# model = SiameseLSTM()
# api.build_graph(model, "/gpu:0")

# api.load_model("latest")

from semantic_model_api import SemanticModel
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

api = SemanticModel(
    model_path="/data/xuht/duplicate_sentence_model/duplicate_models",
    model_list=["BIMPM"]
    )

api.build_model()
api.init_random_model()
api.init_pretrained_model()

question = u"派发红利是发到银行卡上吗？"
candidate_list = [u"请问，分红是打到我的银行还是直接转账呢", 
                    u"发红利是发到银行卡上吗？", 
                    u"本金已缴了多少？分红有多少？", 
                    u"分红是自动到账，还是需要去操作领取", 
                    u"请问如何查看万能账户余额",
                    u"你好！我看到有红利发放，怎么回事儿，不是每年有",
                    u"是年的团险微理赔额度吗",
                    u"请问这个红利是什么？可以取出来吗？"]

# question = u"修改服务策略和设置保留金额什么时候生效"
# candidate_list = [u"上午设置保留金额下午会生效嘛",
#             u"如何查询sim卡的手机号码",
#            u"如何从北京给上海打电话",u"我的保险合同的号码是多少", u"派发红利是发到银行卡上吗？"]

result = api.infer(question, candidate_list, cut_tool)

for res, candidate in zip(result, candidate_list):
    print res, candidate, question
