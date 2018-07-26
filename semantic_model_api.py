import tensorflow as tf
import numpy as np
import pickle as pkl
import jieba

from data_utils import (read_emebdding, read_data, utt2id, 
    corpus_to_vocab, w2v_to_corpus_vocab, id2utt)

from model_api import ModelAPI
from siamese_lstm import SiameseLSTM
from siamese_cnn import SiameseCNN
from bimpm import BIMPM
from bimpm_new import BIMPM_NEW
from match_pyramid import MatchPyramid
from qa_cnn import QACNN
from dan import DAN
from dan_fast import DANFast
from disan import DiSAN
from biblosa import BiBLOSA
from transformer_encoder import TransformerEncoder
from lstm_match_pyramid import LSTMMatchPyramid

from collections import OrderedDict
import os, json, codecs, sys

class SemanticModel(object):
    def __init__(self):
        print("------------using tf-based semantic model-----------")

    def init_config(self,  config):
        self.model_path = config.get("model_path",None)
        self.model_list = config.get("model_list", None)
        self.model_config_path = config.get("model_config_path", None)
        if self.model_config_path is not None:
            if sys.version_info < (3, ):
                self.model_config = pkl.load(open(self.model_config_path, "rb"))
            else:
                self.model_config = pkl.load(open(self.model_config_path, "rb"), encoding="iso-8859-1")
            print("----succeeded in reading model config pkl-----")
        else:

            self.model_config = {
                # "embed_path":self.model_path,
                # "SiameseLSTM_l1":{
                #     "model":SiameseLSTM(),
                #     "model_config_path":os.path.join(self.model_path, "siamese_lstm_l1_contrastive_metric"),
                #     "embed_path":os.path.join(self.model_path, "siamese_lstm_l1_contrastive_metric"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:0",
                #                     "model_id":"18000"}
                # },
                "SiameseLSTM_l2":{
                    "model":SiameseLSTM(),
                    "model_config_path":os.path.join(self.model_path, "siamese_lstm_l2_contrastive_metric"),
                    "embed_path":os.path.join(self.model_path, "siamese_lstm_l2_contrastive_metric"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:0",
                                    "model_id":"21000"}
                },
                "SiameseCNN":{
                    "model":SiameseCNN(),
                    "model_config_path":os.path.join(self.model_path, "siamese_cnn"),
                    "embed_path":os.path.join(self.model_path, "siamese_cnn"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:0",
                                    "model_id":"1200"}
                },
                # "BIMPM":{
                #     "model":BIMPM(),
                #     "model_config_path":os.path.join(self.model_path, "bimpm"),
                #     "embed_path":os.path.join(self.model_path, "bimpm"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:0",
                #                     "model_id":"4800"}
                # },
                # "BIMPM_focal_loss":{
                #     "model":BIMPM(),
                #     "model_config_path":os.path.join(self.model_path, "bimpm_focal_loss"),
                #     "embed_path":os.path.join(self.model_path, "bimpm_focal_loss"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:0"}
                # },
                "MatchPyramid":{
                    "model":MatchPyramid(),
                    "model_config_path":os.path.join(self.model_path, "match_pyramid"),
                    "embed_path":os.path.join(self.model_path, "match_pyramid"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:0",
                                    "model_id":"18000"}
                },
                # "MatchPyramid_focal_loss":{
                #     "model":MatchPyramid(),
                #     "model_config_path":os.path.join(self.model_path, "match_pyramid_focal_loss"),
                #     "embed_path":os.path.join(self.model_path, "match_pyramid_focal_loss"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:0"}
                # },
                # "DiSAN":{
                #     "model":DiSAN(),
                #     "model_config_path":os.path.join(self.model_path, "disan"),
                #     "embed_path":os.path.join(self.model_path, "disan"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:0",
                #                     "model_id":"72000"}
                # },
                # "BIMPM_NEW":{
                #     "model":BIMPM_NEW(),
                #     "model_config_path":os.path.join(self.model_path, "bimpm_new"),
                #     "embed_path":os.path.join(self.model_path, "bimpm_new"),
                #     "updated_config":{"gpu_ratio":0.9, 
                #                     "device":"/gpu:2",
                #                     "model_id":"9000"}
                # },
                "QACNN":{
                    "model":QACNN(),
                    "model_config_path":os.path.join(self.model_path, "qacnn"),
                    "embed_path":os.path.join(self.model_path, "qacnn"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:2",
                                    "model_id":"14400"}
                },
                "DAN":{
                    "model":DAN(),
                    "model_config_path":os.path.join(self.model_path, "dan"),
                    "embed_path":os.path.join(self.model_path, "dan"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:2",
                                    "model_id":"14400"}
                },
                "BiBLOSA":{
                    "model":BiBLOSA(),
                    "model_config_path":os.path.join(self.model_path, "biblosa"),
                    "embed_path":os.path.join(self.model_path, "biblosa"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:2",
                                    "model_id":"14400"}
                },
                "BiBLOSA_DiSAN":{
                    "model":BiBLOSA(),
                    "model_config_path":os.path.join(self.model_path, "biblosa_disa"),
                    "embed_path":os.path.join(self.model_path, "biblosa_disa"),
                    "updated_config":{"gpu_ratio":0.9, 
                                    "device":"/gpu:2",
                                    "model_id":"14400"}
                },
                "TransformerEncoder":{
                    "model":TransformerEncoder(),
                    "model_config_path":os.path.join(self.model_path, "transformer_encoder"),
                    "embed_path":os.path.join(self.model_path, "transformer_encoder"),
                    "updated_config":{
                        "gpu_ratio":0.9,
                        "device":"/gpu:2",
                        "model_id":"14000"
                    }
                },
                "DANFast":{
                    "model":DANFast(),
                    "model_config_path":os.path.join(self.model_path, "dan_fast"),
                    "embed_path":os.path.join(self.model_path, "dan_fast"),
                    "updated_config":{
                        "gpu_ratio":0.9,
                        "device":"/gpu:2",
                        "model_id":"14000"
                    }
                },
                "LSTMMatchPyramid":{
                    "model":LSTMMatchPyramid(),
                    "model_config_path":os.path.join(self.model_path, "lstm_match_pyramid"),
                    "embed_path":os.path.join(self.model_path, "lstm_match_pyramid"),
                    "updated_config":{
                        "gpu_ratio":0.9,
                        "device":"/gpu:2",
                        "model_id":"14000"
                    }
                }
            }
            pkl.dump(self.model_config, open(os.path.join(self.model_path, "semantic_model_config_new.pkl"), "wb"))
     
    def add_model(self, model, model_name):
        self.model_config[model_name] = {
            "model":model,
            "model_config_path":os.path.join(self.model_path, model_name),
            "embed_path":os.path.join(self.model_path, model_name),
            "updated_config":{"gpu_ratio":0.9, 
                            "device":"/gpu:2"}
        }
        
    def build_model(self):
        self.model_api = OrderedDict()
        for model_type in self.model_list:
            if model_type in self.model_config.keys():
                if not os.path.exists(self.model_config[model_type]["model_config_path"]):
                    os.path.mkdir(self.model_config[model_type]["model_config_path"])
                self.model_api[model_type] = ModelAPI(self.model_config[model_type]["model_config_path"], 
                                         self.model_config[model_type]["embed_path"])
                self.model_api[model_type].model_type = model_type
                print(self.model_api[model_type].model_type,"-----model type----")
                self.model_api[model_type].load_config()
                self.model_api[model_type].update_config(self.model_config[model_type]["updated_config"])

    def init_random_model(self):
        device_list = [self.model_config[model_type]["updated_config"]["device"] for model_type in self.model_list if model_type in self.model_config]
        for model_type, device in zip(list(self.model_api.keys()), device_list):
            print(model_type, device)
            self.model_api[model_type].build_graph(self.model_config[model_type]["model"],
                                                    device)
          
    def init_pretrained_model(self, mode="latest"):
        for model_type in list(self.model_api.keys()):
            self.model_api[model_type].load_model(mode)
            print("----------------Succeeded in restoring latest stored model---------------------", model_type)
    
    def build_infer_batch(self, question, candidate_list, max_length, token2id, cut_tool):
        question_utt2id = [utt2id(" ".join(cut_tool.cut(question)), token2id, max_length) for _ in range(len(candidate_list))]
        candidate_utt2id = [utt2id(" ".join(cut_tool.cut(item)), token2id, max_length) for item in candidate_list]

        anchor_matrix = np.asarray(question_utt2id).astype(np.int32)
        check_matrix = np.asarray(candidate_utt2id).astype(np.int32)

        question_len_matrix = np.sum(anchor_matrix > 0, axis=-1)
        check_len_matrix = np.sum(check_matrix > 0, axis=-1)

        return [anchor_matrix, check_matrix, "", question_len_matrix, check_len_matrix]
    
    def infer_batch_features(self, infer_batch):
        feature_matrix = []
        for index, model_type in enumerate(list(self.model_api.keys())):
            features = self.model_api[model_type].infer_features(infer_batch)
            if index == 0:
                feature_matrix = features
            else:
                feature_matrix = np.hstack((feature_matrix, features))

        return feature_matrix

    def infer_ensemble_features(self, question, candidate_list, cut_tool):
        model_type = list(self.model_api.keys())[0]
        max_length = self.model_api[model_type].config["max_length"]
        token2id = self.model_api[model_type].config["token2id"]

        infer_batch = self.build_infer_batch(question, candidate_list, 
                                            max_length, token2id, cut_tool)

        feature_matrix = self.infer_batch_features(infer_batch)
        return feature_matrix, infer_batch

    def infer_predict_probs(self, infer_batch):
        total_score_list = []
        for model_type in list(self.model_api.keys()):
            [logits, pred_probs] = self.model_api[model_type].infer_step(infer_batch)
            similar_preds = list(pred_probs[:,-1])
            total_score_list.append(similar_preds)

        similar_score_matrix = np.stack(total_score_list, axis=1)
        batch_size = similar_score_matrix.shape[0]
        similar_score_matrix = similar_score_matrix.reshape((batch_size, -1))
        return similar_score_matrix

    def voting(self, infer_batch):
        total_score_list = []
        for model_type in list(self.model_api.keys()):
            [logits, pred_probs] = self.model_api[model_type].infer_step(infer_batch)
            similar_preds = list(pred_probs[:,-1])
            print(model_type, similar_preds)
            total_score_list.append(similar_preds)

        similar_score_matrix = np.asarray(total_score_list).astype(np.float32)
        similar_score_matrix = np.transpose(similar_score_matrix, (1,0))

        threshold_matrix = similar_score_matrix >= 0.8
        hit_matrix = np.sum(threshold_matrix, axis=-1)
        
        max_scores = np.max(similar_score_matrix, axis=-1)

        return [(hit_num, max_score) for hit_num, max_score in zip(hit_matrix, max_scores)]

    def infer(self, question, candidate_list, cut_tool):
        model_type = list(self.model_api.keys())[0]
        max_length = self.model_api[model_type].config["max_length"]
        token2id = self.model_api[model_type].config["token2id"]

        infer_batch = self.build_infer_batch(question, candidate_list, 
                                            max_length, token2id, cut_tool)

        output_score = self.voting(infer_batch)
        return output_score

    def output_score(self, question, candidate_list, cut_tool):
        model_type = list(self.model_api.keys())[0]
        max_length = self.model_api[model_type].config["max_length"]
        token2id = self.model_api[model_type].config["token2id"]

        infer_batch = self.build_infer_batch(question, candidate_list, 
                                            max_length, token2id, cut_tool)
        total_score_list = []
        for model_type in list(self.model_api.keys()):
            [logits, pred_probs] = self.model_api[model_type].infer_step(infer_batch)
            similar_preds = list(pred_probs[:,-1])
            print(model_type, similar_preds)
            total_score_list.append(similar_preds)
        return total_score_list








    
            
    
                
