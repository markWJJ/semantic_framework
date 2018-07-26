# -*- coding: utf-8 -*- 
from api.import_excel import ImportHandler, IntervalHandler
from api.export_excel import ExportHandler
from elasticsearch import Elasticsearch
from guoxin_poc_test import poc_api
from settings.setting import database_path, word_path, default_port
from tornado.options import define, options
import tornado.options
import tornado.ioloop
import tornado.web
import logging
import os

define("port", default=default_port, help="run on the port", type=int)


ES_URL = os.environ.get("ES_URL", "192.168.3.133")
ES_PORT = os.environ.get("ES_PORT", 9200)


urls = [
        (r"/api/upload", ImportHandler),
        (r"/api/download", ExportHandler),
        (r"/api/interval", IntervalHandler),
    ]


class Application(tornado.web.Application):

    def __init__(self):
        handlers = urls
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            static_path=os.path.join(os.path.dirname(__file__), 'static')
        )
        super(Application, self).__init__(handlers, **settings)
        # Have one global connection to the blog DB across all handlers
        self.es = Elasticsearch(["{}:{}".format(ES_URL, ES_PORT)])
        self.logger = logging.getLogger()
        self.block = '0'
        self.init_index()

    def init_index(self):
        data={
                "index_es":False,
                "store_es":False,
                "retrive_es":False,
                "delete_es":True,
                "es_api":self.es,
                "database_index":"guoxin_es_database",
                "semantic_index":"guoxin_semantic_result",
                "entity_path": "/data/xuht/guoxin/poc/stock_entities.txt",
                "es_database": "/data/xuht/guoxin/poc/poc_total_clean_data.json",
                "query": u"基金如何购买",
                "data_id":0,
            }
        poc_api(data)
        data["index_es"] = True
        data["delete_es"] = False
        poc_api(data)


def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application(), xheaders=True)
    http_server.bind(options.port)
    http_server.start()
    print('server start')
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    print("-----start server -----------")
    main()