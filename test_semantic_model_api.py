# -*- coding: utf-8 -*-

from semantic_model_api import SemanticModel
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

api = SemanticModel(
	model_path="/data/xuht/duplicate_sentence_model/duplicate_models",
	model_list=["SiameseLSTM_l1", "SiameseLSTM_l2", "SiameseCNN", "BIMPM", "MatchPyramid", "DiSAN"]
	)

api.build_model()
api.init_random_model()
api.init_pretrained_model()

# result = api.infer(question, candidate_list)

# for res, candidate in zip(result, candidate_list):
# 	print res, candidate, question

from flask import Flask, render_template,request,json
from flask import jsonify
import json
import flask
from collections import OrderedDict
import requests
from pprint import pprint

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

# elastic_search_api = Elasticsearch([es_url])
# elastic_search_api

def search(data):
	results = api.infer(data["query"], data["database"])
	sorted_index = sorted(range(len(results)), 
            key=lambda k: (results[k][0], results[k][1]), reverse=True)
	return sorted_index

def output_answers(data):
	return [data["database"][index] for index in data["candidate_index"]]

def add2es(input_data, es_api, database_key, doc_type, index_key):
	input_data["question"] = " ".join(list(jieba.cut(input_data["question"])))
	es_api.index(index=database_key, doc_type=doc_type, id=index_key, body=input_data)

def search_es(input_data, id_type, es_api, database_key):
	body = {
            "query": 
                {
                    "match":{id_type:" ".join(list(jieba.cut(input_data)))}           
                }
          }
	results = es_api.search(index=database_key, body=body)
	outputs = [item["_source"] for item in results['hits']['hits'][0:20]]
	return outputs

def parser(data):
	t = data.get("qa")
    
	search_cond = {
        "query": {
            "match": {
                "query_question": data["query"],
            }
        }
     }
	url = "http://{}:{}/{}/_search".format(t[0], t[1], t[2])
	r = requests.get(url=url, data=json.dumps(search_cond))
	results = json.loads(r.content.decode())
	es_results = [item["_source"] for item in results['hits']['hits'][0:15]]
	database = format_es_data({"es_data":es_results})
	print("----------database-----------", database)
	sorted_index = search({"database":database, "query":data["query"]})
	output_info = output_answers({"database":es_results, "candidate_index":sorted_index})
	return output_info
    
def format_es_data(data):
	database = []
	for item in data["es_data"]:
		database.append(item["query_question"])
	return database



create_api([parser], 8801)