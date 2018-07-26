# -*- coding: utf-8 -*- 
import tornado.web
import json


class BaseHandler(tornado.web.RequestHandler):

    def json_dumps(self, obj):
        # return json.dumps(obj, default=alchemyencoder,
        # ensure_ascii=False).encode("utf-8")
        return json.dumps(obj, ensure_ascii=False)

    def response(self, result):
        self.write(self.json_dumps(result))

    def set_block(self, value='0'):
        self.application.block = value

    def get_block(self):
        return self.application.block

    @property
    def es(self):
        e = self.application.es
        return e