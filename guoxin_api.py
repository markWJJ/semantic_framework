# -*- coding: utf-8 -*-
from data_cleaner import DataCleaner

import jieba, re, codecs
import jieba.posseg as pseg

from pprint import pprint
import itertools, sys

from collections import OrderedDict
from fuzzywuzzy import fuzz

from semantic_api import SemanticAPI
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
api = SemanticAPI()
api.init_model("wmd")
api.init_model("tf_model")
# api.init_model("ensemble")
api.init_model("token_distance")

import json
import codecs
import os
import sys
from data_utils import cut_tool_api
from acora import AcoraBuilder

import logging

cut_tool = cut_tool_api()
data_cleaner_api = DataCleaner({
        "stop_word":"/data/xuht/guoxin/poc/stop_words.txt",
        "synthom_path":"/data/xuht/guoxin/poc/synthoms.txt",
        "non_words":"/data/xuht/guoxin/poc/no_words.txt"
})

from elasticsearch import helpers

def delete_es(data):
    try:
        data["es_api"].indices.delete(index=data["database_index"])
        print("-----already delete es database-----", data["database_index"])
        flag_database = True
    except:
        print("-----not exist es database-----", data["database_index"])
        flag_database = False

    try:
        data["es_api"].indices.delete(index=data["semantic_index"])
        print("-------already delete es index----", data["semantic_index"])
        flag_database = True
    except:
        print("-------not exist es index----", data["semantic_index"])
        flag_database = False
    return flag_database

def get_es_results(data):
    es_client = data["es_api"]
    index = data["semantic_index"]
    es_search_options = set_search_optional()
    es_result = get_search_result(es_client, es_search_options, 
                                    index=index, doc_type=index)
    final_result = get_result_list(es_result)
    return final_result

def get_result_list(es_result):
    final_result = []
    for item in es_result:
        final_result.append(item['_source'])
    return final_result

def get_search_result(es_client, es_search_options, scroll="60s", 
                      index='guoxin_semantic_result', 
                      doc_type='guoxin_semantic_result', 
                      timeout="60s"):
    es_result = helpers.scan(
        client=es_client,
        query=es_search_options,
        scroll=scroll,
        index=index,
        doc_type=doc_type,
        timeout=timeout
    )
    return es_result

def set_search_optional():
    # 检索选项
    body = {
            "query":
                    {
                        "match": {"results":"ok"}
                    }
            }
    
    es_search_options = body
    return es_search_options

def es_index_doc(data):
    import codecs, json
    es_api = data["es_api"]
    es_index = data["database_index"]
    es_database = json.load(codecs.open(data["es_database"], "r", "utf-8"))
    idx = 0
    entity_path = data.get("entity_path", None)
    print("----entity path----", entity_path)
    if entity_path:
        load_jieba_userdict(entity_path)
    index = 0
    for item in es_database:
        if data.get("re_ner", False):
            tmp = {}
            question = item["origin"].lower()
            template, entity_list = extract_entity(question)
            for key in entity_list:
                tmp[key] = " ".join(entity_list[key])
            item["entity"] = tmp
            item["template"] = template

        data["es_api"].index(index=es_index, 
                    doc_type=es_index,
                    body=item,
                    id=item["id"],
                    timeout="60s")
        index += 1
    print("------succeeded in indexing es------", index)

def preprocess(input_text):
    tmp = data_cleaner_api.poc_clean(input_text)
    return tmp

# def get_key_word(data):
#     output_database = []
#     if len(data["entity_dict"]) >= 1:
#         dicts = OrderedDict()
#         for key in data["entity_dict"]:
#             dicts[key] = key
#             for t in data["entity_dict"][key]:
#                 dicts[t] = key
#         query = data["query"]
#         key_word_builder = AcoraBuilder(dicts.keys())
#         key_word_searcher = key_word_builder.build()
#         print(dicts, "------detected diccts-------")
#         res = key_word_searcher.findall(query)
#         print(res)
#         if len(res) >= 1:
#             input_entity = [item[0] for item in res]
#             input_entity_key = []
#             for char in input_entity:
#                 input_entity_key.extend(data["entity_dict"][dicts[char]])
#                 input_entity_key.append(dicts[char])
#             input_key_entity = list(set(input_entity_key))
#             key_word_builder = AcoraBuilder(input_key_entity)
#             key_word_searcher = key_word_builder.build()
#             for data in data["database"]:
#                 t = len(key_word_searcher.findall(data))
#                 output_database.append(t)
#         else:
#             for data in data["database"]:
#                 output_database.append(0)
#     else:
#         for data in data["database"]:
#             output_database.append(0)
#     return output_database

# def postprocess(data, output_answer):
#     max_answer_similar_question_scores = []
#     input_question = data["query"]
#     for answer in output_answer:
#         data["query"] = answer
#         es_search_body = {"answer":data["query"]}
#         output_database, tmp_output_answer = es_search(data, es_search_body)
#         entity_path = data.get("entity_path", None)
#         if entity_path:
#             with codecs.open(entity_path, "r", "utf-8") as frobj:
#                 entity_lists = frobj.read().splitlines()
#                 entity = extract_entity(data["query"], output_database)
#         else:
#             entity = [0]*len(output_database)
#         candidate_database = {"query":input_question,
#                             "database":output_database,
#                             "entity_hits":entity}
#         sorted_index, results = search(candidate_database)
#         output_info = output_answers({"database":output_database, "candidate_index":sorted_index, "results":results})
#         max_answer_similar_question_scores.append([output_info[0][1], answer])
#     s = sorted(max_answer_similar_question_scores, key=lambda x:x[0], reverse=True)
#     print(s)
#     return s[0][1]

def load_jieba_userdict(user_dict_path):

    with codecs.open(user_dict_path, "r", "utf-8") as frobj:
        lines = frobj.read().splitlines()
        for line in lines:
            content = line.split()
            key_entity = data_cleaner_api.full2half(content[0])
            key_entity = data_cleaner_api.tra2sim(key_entity.lower())
            jieba.add_word(key_entity, int(content[1]), content[2])
    print(pseg.lcut(u"麒麟家居"))

def set_cnt(string1, string2, cnt, key):
    set1 = list(set(string1.split()))
    set2 = list(set(string2.split()))
    if key == "<stock_name>" or key == "<stock_id>":
        if len(set1) >= 1 and len(set2) >= 1:
            cnt += 1
    else:
        if len(set1) >= 1 and len(set2) >=1:
            for item in set1:
                if item in set2:
                    cnt += 1
    return cnt

def get_entity_number(query_entity, output):
    entity_hits = [0]*len(output)
    for index, item in enumerate(output):
        cnt = 0
        for key in ["<key_entity>", "<stock_id>", "<stock_name>"]:
            cnt = set_cnt(query_entity[key], item["entity"][key], cnt, key)
        entity_hits[index] = cnt
            # if len(query_entity[key]) >= 1:
            #     if query_entity[key] == item["entity"][key]:
            #         print query_entity[key], item["entity"][key],index, key
            #         entity_hits[index] += 1
            #     elif key == "<stock_id>" or "<stock_name>":
            #         if len(query_entity[key]) >= 1 and len(item["entity"][key]) >= 1:
            #             print query_entity[key], item["entity"][key],index, key
            #             entity_hits[index] += 1
            #     else:
            #         if len(item["entity"][key]) >= 1:
            #             if query_entity[key] in item["entity"][key] or item["entity"][key] in query_entity[key]:
            #                 print query_entity[key], item["entity"][key],index, key
            #                 entity_hits[index] += 1
                # else:
                #     sub_entity = query_entity[key].split()
                #     if len(sub_entity) >= 2:
                #         for t in sub_entity:
                #             if t in item["entity"][key] or t == item["entity"][key]:
                #                 entity_hits[index] += 0.1
    return entity_hits

def search_answer(data):
    search_body = {
            "query":{
                "match":{
                    "answer":data["query"]
                }
            }
    }
    output = es_search(data, search_body)
    return output

def get_non_word(input_list):
    non_word_cnt = []
    for word in input_list:
        non_word_cnt.append(data_cleaner_api.calculate_non_word(word))
    return non_word_cnt

def strcuture_infer(input_string):
    cut_index = -1
    for index, word in enumerate(list(input_string)):
        if word == u"转":
            cut_index = index

    if cut_index == -1:
        return 1

    bank_index = 0
    warrant_index = 0
    direction_index = 0
    
    for index, word in enumerate(list(input_string)):
        if word == u"银":
            bank_index = index
        elif word == u"证" or word == u"券":
            warrant_index = index
        elif word == u"到" or word == u"入" or word == u"进":
            if index > cut_index:
                direction_index = 1
        elif word == u"出":
            if index > cut_index:
                direction_index = -1

    if bank_index < cut_index and cut_index < warrant_index:
        return 1
    elif bank_index > cut_index and cut_index > warrant_index:
        return 0
    elif bank_index == 0 or warrant_index == 0:
        if bank_index == 0:
            if direction_index == 0:
                if cut_index < warrant_index:
                    return 1
                else:
                    return 0
            else:
                if cut_index < warrant_index:
                    if direction_index == 1:
                        return 1
                    else:
                        return 0
                else:
                    if direction_index == 1:
                        return 0
                    else:
                        return 1
        elif warrant_index == 0:
            if direction_index == 0:
                if cut_index < bank_index:
                    return 0
                else:
                    return 1
            else:
                if cut_index < bank_index:
                    if direction_index == 1:
                        return 0
                    else:
                        return 1
                else:
                    if direction_index == 1:
                        return 1
                    else:
                        return 0
        else:
            return 0
    else:
        return 0

def semantic_match_api(data):
    preprocess_query = preprocess(data["query"])
    real_question = data["query"]
    print(preprocess_query)
    data["query"] = preprocess_query

    entity_path = data.get("entity_path", None)
    if entity_path:
        load_jieba_userdict(data["entity_path"])

    template, result = extract_entity(preprocess_query)
    new_result = {"origin":preprocess_query, "template":template}
    for key in result:
        new_result[key] = " ".join(result[key])  

    output = get_es_candidate(data, new_result, 
        ["<key_entity>", "<stock_id>", "<stock_name>"])

    logging.info("-------------check es results {}".format(output))

    query_non_word_cnt = data_cleaner_api.calculate_non_word(preprocess_query)
    query_structure = strcuture_infer(preprocess_query)

    output_database = []
    output_answer = []
    filtered_output = []

    if u"转" in preprocess_query and u"银" in preprocess_query or u"证" in preprocess_query or u"券" in preprocess_query:
        for index, item in enumerate(output):
            query = preprocess(item["origin"])
            candidate_non_word_cnt = data_cleaner_api.calculate_non_word(query)
            candidate_structure = strcuture_infer(query)
            
            if candidate_non_word_cnt == query_non_word_cnt and query_structure == candidate_structure:
                if len(query) >= 1:
                    output_answer.append(item["answer"])
                    output_database.append(query)
                    filtered_output.append(item)
        if len(filtered_output) >= 1:
            output = filtered_output
        else:
            for index, item in enumerate(output):
                query = preprocess(item["origin"])
                if len(query) >= 1:
                    output_answer.append(item["answer"])
                    output_database.append(query)
                    filtered_output.append(item)
            if len(filtered_output) >= 1:
                output = filtered_output
    else:
        for index, item in enumerate(output):
            query = preprocess(item["origin"])
            if len(query) >= 1:
                output_answer.append(item["answer"])
                output_database.append(query)
                filtered_output.append(item)

        if len(filtered_output) >= 1:
            output = filtered_output

    entity_hits = get_entity_number(new_result, output)

    for q, entity in zip(output_database, entity_hits):
        if sys.version_info < (3, ):
            print( q, entity, preprocess_query)
        else:
            logging.info("-------------sorted_index {} {} {}".format(q, entity, preprocess_query))

    candidate_database = {"query":preprocess_query,
                            "database":output_database,
                            "entity_hits":entity_hits}

    matched_question = real_question
    if len(output_database) == 1:
        sorted_index = [0]
        most_revelant_answer = output_answer[0]
        matched_question = output_database[0]
    elif len(output_database) == 0 or len(data["query"]) == 0:
        sorted_index = [0]
        most_revelant_answer = u"亲~请您输入9转在线人工服务哦，人工服务时间为交易日8：30-17：00. 小信机器人是否解决了您的问题，请输入数字:【9】未解决,转人工【0】已解决直接输入问题可继续咨询。"
        matched_question = real_question
    else:
        try:
            model_type = data.get("model_type", "voting")
            sorted_index, results = search(candidate_database, model_type)
            logging.info("-------------check match api {}".format(sorted_index))
            output_info = output_answers({"database":output_database, "candidate_index":sorted_index, "results":results})
            most_revelant_answer = output_answer[sorted_index[0]]
            try:
                matched_question = output_database[sorted_index[0]]
            except:
                matched_question = real_question
        except:
            sorted_index = [0]
            matched_question = real_question
            most_revelant_answer = u"亲~请您输入9转在线人工服务哦，人工服务时间为交易日8：30-17：00. 小信机器人是否解决了您的问题，请输入数字:【9】未解决,转人工【0】已解决直接输入问题可继续咨询。"

    logging.info("-------------sorted_index {}".format(sorted_index))
    print("------------------------", sorted_index)

    # for index in range(len(sorted_index)):
    #     print preprocess_query, output_database[sorted_index[index]], output_answer[sorted_index[index]]
    if data.get("debug", False) == True:
        intermediate_result_store(data, 
                                most_revelant_answer,
                                output,
                                sorted_index,
                                real_question,
                                new_result,
                                data["data_id"],
                                matched_question)
        return output, sorted_index
    else:
        intermediate_result_store(data, 
                                most_revelant_answer,
                                "",
                                sorted_index,
                                real_question,
                                "",
                                data["data_id"],
                                matched_question)
        return output

def intermediate_result_store(data, output_answer, output, 
                            sorted_index, query, new_result, data_id, 
                            matched_question):
    tmp = {"answer":output_answer, "question":query, 
            "id":data_id, "results":"ok", "database":output, 
            "sorted_index":sorted_index, "entity":new_result, 
            "matched_question":matched_question}
    es_api = data["es_api"]
    index = data["semantic_index"]
    es_api.index(index=index,
                    doc_type=index,
                    body=tmp,
                    id=tmp["id"],
                    timeout="60s")
    print("-----succeeded in storing result in es 11------")

def retrive_results(data):
    results = get_es_results(data)
    sorted_results = sorted(results, key=lambda x : x["id"])
    return [{"answer":item["answer"], 
            "query":item["question"], 
            "database":item["database"],
            "matched_question":item["matched_question"],
            "sorted_index":item["sorted_index"]} for item in sorted_results]

def search(data, model_type):
    print("-------------model_type--------------", model_type, len(data["database"]))
    candidate = data["database"]
    if len(candidate) < 30:
        candidate.extend([candidate[-1]]*(30-len(candidate)))
    elif len(candidate) > 30:
        candidate = data["database"][0:30]
    else:
        candidate = data["database"]
    assert len(candidate) == 30
    results = api.infer(data["query"], candidate, model_type)
    print("-------results---------", results)
    all_results = []
    cnt = 0
    for index in range(len( data["entity_hits"])):
        entity = data["entity_hits"][index]
        item = results[index]
        all_results.append((entity, item[0], item[1]))
            
    print(all_results,"--------entity new results-------", len(all_results), len(data["entity_hits"]))
    assert len(all_results) == len(data["entity_hits"])
    hitted_results = []
    left_results = []
    for item in all_results:
        if item[1] >= 1 or item[-1] >= 0.85:
            hitted_results.append(item)
        else:
            left_results.append(item)

    sorted_index = sorted(range(len(hitted_results)), 
        key=lambda k: (hitted_results[k][0], hitted_results[k][1], hitted_results[k][2]), reverse=True)
    result_sorted_index = sorted_index + [index + len(hitted_results) for index in range(len(left_results))]
    return sorted_index, hitted_results+left_results

def output_answers(data):
    print(data["results"])
    ret = [(data["database"][item], str(data["results"][item][-1])) for item in data["candidate_index"]]

    print(ret[0][0], ret[0][1])
    return ret

def construct_search_body(new_result, key_list, mode=1):
    filter_body = []
    should_body = []
    must_body = []
    for key in key_list:
        if len(new_result[key]) >= 1:
            for item in new_result[key].split():
                filter_body.append({"match":{"entity."+key:item}})
            #filter_body.append({"match":{"entity."+key:new_result[key]}})
    must_body.append({"match":{"origin":new_result["origin"]}})
    should_body.append({"match":{"template":new_result["template"]}})
    if mode == 0:
        body = {
            "query":{
                "bool":{
                    "filter":filter_body,
                    "must":must_body,
                }
            }
        }
        if len(should_body) >= 1:
            body["query"]["bool"]["should"] = should_body

    elif mode == 1:
        body = {
            "query":{
                "bool":{
                    "should":should_body,
                    "must":must_body,
                }
            }
        }
        if len(filter_body) >= 1:
            body["query"]["bool"]["must"] += filter_body
        
    elif mode == 2:
        body = {
            "query":{
                "bool":{
                    "must":must_body,
                }
            }
        }
    from pprint import pprint
    pprint(body)
    return body

def es_search(data, search_body):
    es_api = data["es_api"]
    index = data["database_index"]
    results = es_api.search(index=index, body=search_body, size=20, timeout="30s")
    results = results["hits"]["hits"]
    output = []
    if len(results) >= 1:
        max_score = results[0]["_score"]
        scores = [(index, item["_score"] / max_score) for index,item in enumerate(results)]
        for score in scores:
            if score[1] > 0.6:
                output.append(results[score[0]]["_source"])
    return output

def get_combination_index(input_list):
    output_combination_index = []
    for index in range(len(input_list), -1, -1):
        list_combination = list(itertools.combinations(input_list, index+1))
        for item in list_combination:   
            output_combination_index.append(list(item))
    return output_combination_index

def extract_entity(input_sentence):
    
    output_entity_list = {"<stock_id>":[], "<stock_name>":[], "<key_entity>":[]}
    tmp_txt = input_sentence
                
    words = pseg.lcut(tmp_txt)
    replace_name = {"<stock_name>":"AAA", "<key_entity>":"BBB"}
    reverse_name = {"AAA":"<stock_name>", "BBB":"<key_entity>"}
    
    for word in words:
        tmp_word = list(word)
        if tmp_word[1] in ["<stock_name>", "<key_entity>"]:
            if tmp_word[1] in output_entity_list:
                tmp_txt = re.sub(tmp_word[0], replace_name[tmp_word[1]], tmp_txt)
                output_entity_list[tmp_word[1]].append(tmp_word[0])
            else:
                tmp_txt = re.sub(tmp_word[0], replace_name[tmp_word[1]], tmp_txt)
                output_entity_list[tmp_word[1]] = [tmp_word[0]]
                
    stock_id_pattern = re.compile("[a-z0-9]{6}")
    stock_id_result = stock_id_pattern.findall(tmp_txt)
    if len(stock_id_result) > 0:
        for result in stock_id_result:
            if "<stock_id>" in output_entity_list:
                output_entity_list["<stock_id>"].append(result)
            else:
                output_entity_list["<stock_id>"] = [result]
                
    for key in output_entity_list:
        for item in output_entity_list[key]:
            tmp_txt = re.sub(item, key, tmp_txt)
    for key in reverse_name:
        tmp_txt = re.sub(key, reverse_name[key], tmp_txt)

    return tmp_txt, output_entity_list

def get_es_candidate(data, new_result, data_key_list):
    key_list = [key for key in data_key_list if len(new_result[key]) >= 1]
    if len(key_list) == 0:
        search_body = construct_search_body(new_result, key_list, 2)
        total_output = es_search(data, search_body)
        print("------use loose rearch1--")
    else:
        combination_condition = get_combination_index(key_list)
        total_output = []
        for item in combination_condition:
            search_body = construct_search_body(new_result, item)
            output = es_search(data, search_body)
            print("------use loose rearch3--")
            if len(output) >= 1:
                total_output.extend(output)
        if len(total_output) <= 15:
            print("------use loose rearch4--")
            search_body = construct_search_body(new_result, [])
            output = es_search(data, search_body)
            if len(output) >= 5:
                total_output.extend(output[0:5])
            else:
                total_output.extend(output)
            if len(total_output) <= 20:
                search_body = construct_search_body(new_result, [], 2)
                output = es_search(data, search_body)
                total_output.extend(output)
            else:
                total_output.extend(output)

    output_question = []
    postprocess_output = []
    for item in total_output:
        tmp = preprocess(item["origin"])
        if tmp in output_question:
            continue
        else:
            output_question.append(tmp)
            postprocess_output.append(item)
    if len(postprocess_output) >= 20:
        postprocess_output = postprocess_output[0:20]
    return postprocess_output


    