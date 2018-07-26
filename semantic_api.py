from semantic_model_api import SemanticModel
from data_utils import cut_tool_api, jieba_api
from collections import OrderedDict
from wmd_score_api import WMDAPI
from ensemble_api import EnsembleAPI
import numpy as np
from token_distance_api import TokenDistanceAPI


CONFIG = {
    "tf_model":{
        "model":SemanticModel(),
        "config":{
            #"model_path":"/data/xuht/duplicate_sentence_model/duplicate_models",
            "model_path":"/data/xuht/guoxin/poc/duplicate_sentence_model/duplicate_models",
            "model_list":[
                            "SiameseLSTM_l2",
                            "MatchPyramid", 
                            "QACNN",
                            "DAN",
                            "LSTMMatchPyramid"
                            #"BiBLOSA",
                            #"DANFast",
                            #"TransformerEncoder"
                        ]
            #"model_config_path":"/data/xuht/duplicate_sentence_model/duplicate_models/semantic_model_config.pkl"
            # "BIMPM", 
            # "BIMPM_NEW",
            # "DiSAN"]#, "BIMPM", "BIMPM_NEW"]
        },
        "tool":{
            "cut_tool":
            {
                "model":cut_tool_api(),
                "config":{
                    "user_dict":""
                }
            }
        }
        
    },
    "wmd":{
        "model":WMDAPI(),
        "config":{
            "w2v_path":"/data/xuht/word2vec_model/word2vec_from_weixin/word2vec/wx_vector_char.pkl"
        },
        "tool":{
            "cut_tool":
            {
                "model":cut_tool_api(),
                "config":{
                    "user_dict":"/data/xuht/word2vec_model/word2vec_from_weixin/word2vec/wx_vector_dict.txt"
                }
            }
        }
    },
    "ensemble":{
        "model":EnsembleAPI(),
        "tool":{
            "cut_tool":
            {
                "model":cut_tool_api(),
                "config":{
                    "user_dict":""
                }
            }
        },
        "config":{
            "score_fusion":{
                "model_path":"/data/xuht/guoxin/guoxin_lr_fusion/lr_model.pkl"
            },
            "feature_fusion":{
                "model_path":"/data/xuht/guoxin/guoxin_gbdt_fusion/gbdt_model.pkl"
            }
        }
    },
    "token_distance":{
        "model":TokenDistanceAPI(),
        "config":{},
        "tool":{
            "cut_tool":
            {
                "model":cut_tool_api(),
                "config":{
                    "user_dict":""
                }
            }
        }
    }
}

class SemanticAPI(object):
    def __init__(self, config=None):
        self.config = config if config else CONFIG
        self.api = OrderedDict()
        self.tool = OrderedDict()

    def add_model(self, config):
        model_name = config["model_name"]
        self.config[model_name] = config["model"]

    def update_config(self, updated_dict):
        for key in updated_dict:
            if key in self.config:
                self.config[key]["config"] = updated_dict[key]["config"]
                self.config[key]["tool"] = updated_dict[key]["tool"]

    def init_model(self, model_type):
        self.api[model_type] = self.config[model_type]["model"]
        self.api[model_type].init_config(self.config[model_type]["config"])

        self.api[model_type].build_model()
        self.api[model_type].init_random_model()
        self.api[model_type].init_pretrained_model()
        self.tool[model_type] = OrderedDict()
        for key in self.config[model_type]["tool"]:
            self.tool[model_type][key] = self.config[model_type]["tool"][key]["model"]
            self.tool[model_type][key].init_config(self.config[model_type]["tool"][key]["config"])
            self.tool[model_type][key].build_tool()

    def ensemble_score_infer(self, question, candidate_list):
        [feature_matrix,
        infer_batch] = self.api["tf_model"].infer_ensemble_features(question, 
                                candidate_list, 
                                self.tool["tf_model"]["cut_tool"])

        score_matrix = self.api["tf_model"].infer_predict_probs(infer_batch)

        ensemble_key = list(self.config["ensemble"]["config"].keys())
        infer_feature_dict = dict(zip(ensemble_key, [score_matrix, feature_matrix]))
        ensemble_socre_list = self.api["ensemble"].output_score(infer_feature_dict)

        return ensemble_socre_list

    def infer_ensemble(self, question, candidate_list):
        total_score_list = []
        for model_type in self.api:
            if model_type == "tf_model":
                continue
            elif model_type == "wmd":
                total_score_list.extend(self.api[model_type].output_score(question, 
                                        candidate_list, 
                                        self.tool[model_type]["cut_tool"]))
            elif model_type == "ensemble":
                total_score_list.extend(self.ensemble_score_infer(question, candidate_list))
        similar_score_matrix = np.asarray(total_score_list).astype(np.float32)
        similar_score_matrix = np.transpose(similar_score_matrix, (1,0))

        threshold_matrix = similar_score_matrix >= 0.8
        hit_matrix = np.sum(threshold_matrix, axis=-1)
        
        max_scores = np.max(similar_score_matrix, axis=-1)

        return [(hit_num, max_score) for hit_num, max_score in zip(hit_matrix, max_scores)]

    def infer_voting(self, question, candidate_list):
        total_score_list = []
        for model_type in self.api:
            if model_type == "ensemble":
                continue
            total_score_list.extend(self.api[model_type].output_score(question, candidate_list, self.tool[model_type]["cut_tool"]))
        print("---total score list----", total_score_list)
        similar_score_matrix = np.asarray(total_score_list).astype(np.float32)
        similar_score_matrix = np.transpose(similar_score_matrix, (1,0))

        threshold_matrix = similar_score_matrix >= 0.8
        hit_matrix = np.sum(threshold_matrix, axis=-1)
        
        max_scores = np.max(similar_score_matrix, axis=-1)

        return [(hit_num, max_score) for hit_num, max_score in zip(hit_matrix, max_scores)]

    def infer(self, question, candidate_list, model_type):
        output = []
        if model_type == "voting":
            output = self.infer_voting(question, candidate_list)
        elif model_type == "ensemble":
            output = self.infer_ensemble(question, candidate_list)
        return output
