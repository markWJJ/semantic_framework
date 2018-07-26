# -*- coding: utf-8 -*-

from semantic_model_api import SemanticModel
import os
from data_utils import (read_emebdding, read_data, 
                    utt2id, corpus_to_vocab, 
                    w2v_to_corpus_vocab, id2utt, initialize_embedding)
import jieba, codecs
import numpy as np
from data_cleaner import DataCleaner
import pickle as pkl
import sys

CONFIG = {
    "score_fusion":{
        "model_path":"/data/xuht/guoxin/guoxin_lr_fusion/lr_model.pkl"
    },
    "feature_fusion":{
        "model_path":"/data/xuht/guoxin/guoxin_gbdt_fusion/gbdt_model.pkl"
    }

}

class EnsembleAPI(object):
    def __init__(self):
        print("------------using tf-based score and feature ensemble model-----------")

    def init_config(self,  config):
        self.config = config

    def build_model(self):
        self.model = {}
        for ensemble_type in self.config:
            if sys.version_info < (3, ):
                self.model[ensemble_type] = pkl.load(open(self.config[ensemble_type]["model_path"], "rb"))
            else:
                self.model[ensemble_type] = pkl.load(open(self.config[ensemble_type]["model_path"], "rb"), encoding="iso-8859-1")

    def init_random_model(self):
        pass

    def init_pretrained_model(self):
        pass

    def output_score(self, input_feature_dict):
        score_list = []
        for ensemble_type in self.config:
            feature = input_feature_dict[ensemble_type]
            ensemble_probs = self.model[ensemble_type].predict_proba(feature)
            similar_preds = list(ensemble_probs[:,-1])
            print(ensemble_type, similar_preds)
            score_list.append(similar_preds)
        return score_list
            
