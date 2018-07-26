# -*- coding: utf-8 -*-

import guoxin_api

"""
data interface:
data={
    "es_api":  #实例化的es
    "es_database": # 存储在本地的es_database路径(json格式),
    "query"： 输入的问句,
    "entity_path": 业务实体词典路径
}
"""

def poc_api(data):
    if data["index_es"]:
        guoxin_api.es_index_doc(data)
        return []
    elif data["store_es"]:
        print("-----search es database----")
        output_info = guoxin_api.semantic_match_api(data)
        return output_info
    elif data["retrive_es"]:
        print("-----retrive es-----")
        output = guoxin_api.retrive_results(data)
        return output
    elif data["delete_es"]:
        flag = guoxin_api.delete_es(data)
        return flag
    else:
        flag = guoxin_api.delete_es(data)
        return flag



