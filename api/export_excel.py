from api.base import BaseHandler
from openpyxl import Workbook
from settings.setting import database_path, word_path
from io import BytesIO
from guoxin_poc_test import poc_api

class ExportHandler(BaseHandler):

    def get(self):
        wb = Workbook()
        ws = wb.active  # worksheet

        data={
            "index_es": False,
            "store_es": False,
            "retrive_es": True,
            "delete_es": False,
            "es_api": self.es,
            "database_index": "guoxin_es_database",
            "semantic_index": "guoxin_semantic_result",
            # "entity_path": "/data/stock_entities.txt",
            # "es_database": "/data/guoxin_es_database.json",
            "query": "",
            "data_id": 0,
        }
        result = poc_api(data)
        ws.title = u"result"

        rows = [["question", "answer", "session_id", "matched_question"]]
        for idx, row in enumerate(result):
            rows.append([row.get("query", ""), row.get("answer", ""), idx, row.get("matched_question", "")])
        for row in rows:
            ws.append(row)
        out = BytesIO()
        wb.save(out)
        out.seek(0)
        self.set_header(
            'Content-type',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.set_header('Content-Disposition',
                        'attachment;filename="result.xlsx"')
        self.write(out.getvalue())