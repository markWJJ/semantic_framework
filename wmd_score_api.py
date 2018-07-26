# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.metrics import euclidean_distances
from pyemd import emd

def get_wmd_distance(w2v_model, d1, d2, verbose=False, max_distance_value=10.0, distance_metric="euclidean"):
    
    d1 = [w for w in d1.split() if w in w2v_model]
    d2 = [w for w in d2.split() if w in w2v_model]

    dictionary = list(set(d1+d2))
    W_ = np.array([w2v_model[w] for w in dictionary if w in w2v_model])
    if distance_metric == "euclidean":
        D_ = euclidean_distances(W_)
        D_ = D_.astype(np.double)
        D_ /= max_distance_value  # just for comparison purposes
        
    n = len(dictionary)
    # count occurences
    v_1 = np.zeros((n))
    v_2 = np.zeros((n))
    keys = dict((e[1], e[0]) for e in enumerate(dictionary))
    n = len(keys)
    for word in d1:
        v_1[keys[word]] += 1
    for word in d2:
        v_2[keys[word]] += 1
    # normalize and add default mass
    vmin = 1e-20
    normalize = lambda P: P/(np.tile(np.sum(P), n)+vmin)
    v_1 = normalize(v_1 + np.max(v_1)*vmin)
    v_2 = normalize(v_2 + np.max(v_2)*vmin)
    # pyemd needs double precision input
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)

    return emd(v_1, v_2, D_)

import pickle as pkl
import codecs, sys
class WMDAPI(object):
    def __init__(self):
        print("-------------using wmd model----------------")

    def init_config(self, config):
        self.config = config
        if sys.version_info < (3, ):
            self.w2v_model = pkl.load(open(self.config["w2v_path"], "rb"))
        else:
            self.w2v_model = pkl.load(open(self.config["w2v_path"], "rb"), encoding="iso-8859-1")
        print("----------w2v size-------", len(self.w2v_model))

    def build_model(self):
        pass

    def init_random_model(self):
        pass

    def init_pretrained_model(self):
        pass

    def output_score(self, query_string, candidate_list, cut_tool):
        output_score_ = []
        cut_query_string = cut_tool.cut(query_string)
        for candidate in candidate_list:
            cut_candidate = cut_tool.cut(candidate)
            score = get_wmd_distance(self.w2v_model, cut_query_string, cut_candidate)
            if 1.0 - score >= 0.9:
                output_score_.append(1.0)
            else:
                output_score_.append(1.0 - score)
        print("wmd", output_score_)
        return [output_score_]