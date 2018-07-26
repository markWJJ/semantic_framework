# -*- coding: utf-8 -*-

from semantic_model_api import SemanticModel
import os, jieba
from data_utils import cut_api
from acora import AcoraBuilder

os.environ["CUDA_VISIBLE_DEVICES"]="0"

api = SemanticModel()
api.init_config({"model_path":"/data/xuht/duplicate_sentence_model/duplicate_models",
    "model_list":["SiameseLSTM_l2", "SiameseCNN", "MatchPyramid", "DiSAN", "BIMPM", "BIMPM_NEW"]})
cut_tool = cut_api("")
api.build_model()
api.init_random_model()
api.init_pretrained_model()
    
question = u"修改服务策略和设置保留金额什么时候生效"
candidate_list = [u"上午设置保留金额下午会生效嘛",
            u"如何查询sim卡的手机号码",
           u"如何从北京给上海打电话",u"我的保险合同的号码是多少", u"派发红利是发到银行卡上吗？"]

result = api.infer(question, candidate_list, cut_tool)

for res, candidate in zip(result, candidate_list):
    print res, candidate, question

from flask import Flask, render_template,request,json
from flask import jsonify
import json
import flask
from collections import OrderedDict
import requests
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def make_api(f):
    def web_api():
        data = request.get_json(force=True)
        return jsonify(f(data))
    return web_api

def create_api(functions, port):
    app = Flask(__name__)
    if  not isinstance(functions,list):
        functions = [functions]
    print(functions)
    for f in functions:
        web_api = make_api(f)
        app.add_url_rule('/' + f.__name__, f.__name__, web_api, methods =['POST'])
        print('a api http://0.0.0.0:%s/%s create' %(port,f.__name__ ))
    app.run(debug=False,host='0.0.0.0',port = port)

import json
import os
import jieba
import sys
from elasticsearch import Elasticsearch

def output_answers(data):
    print(data["results"])
    ret = [(data["database"][item], str(data["results"][item][-1])) for item in data["candidate_index"]]

    print((ret))
    return ret
    #return json.dumps(ret, ensure_ascii=False)

def search(data):
    results = api.infer(data["query"], data["database"], cut_tool)
    print("-------results---------", results)
    all_results = []
    for item, entity in zip(results, data["entity_hits"]):
        all_results.append((entity, item[0], item[1]))
    sorted_index = sorted(range(len(all_results)), 
        key=lambda k: (all_results[k][0], all_results[k][1], all_results[k][2]), reverse=True)
    return sorted_index, results

def get_key_word(data):
    output_database = []
    if len(data["entity_dict"]) >= 1:
        dicts = OrderedDict()
        for key in data["entity_dict"]:
            dicts[key] = key
            for t in data["entity_dict"][key]:
                dicts[t] = key
        query = data["query"]
        key_word_builder = AcoraBuilder(dicts.keys())
        key_word_searcher = key_word_builder.build()
        print(dicts)
        res = key_word_searcher.findall(query)
        if len(res) >= 1:
            input_entity = [item[0] for item in res]
            input_key_entity = list(set(input_entity))
            key_word_builder = AcoraBuilder(input_key_entity)
            key_word_searcher = key_word_builder.build()
            for data in data["database"]:
                t = len(key_word_searcher.findall(data))
                output_database.append(t)
        else:
            for data in data["database"]:
                output_database.append(0)
    else:
        for data in data["database"]:
            output_database.append(0)
    return output_database

def parser(data):
    database = data["database"] # format_es_data({"es_data":data})
    output_database = get_key_word(data)
    print("----------database-----------", database)
    for t in database:
        print t
    sorted_index, results = search({"database":database, "query":data["query"], "entity_hits":output_database})
    print("----------------", data["query"])
    print("--------------sorted index-------", sorted_index)
    output_info = output_answers({"database":database, "candidate_index":sorted_index, "results":results})
    return output_info

create_api([parser], 8890)
