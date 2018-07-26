from fuzzywuzzy import fuzz


class TokenDistanceAPI(object):
    def __init__(self):
        print("----------initializing fuzzy distance api--------")

    def init_config(self,  config):
        self.config = config

    def build_model(self):
        pass

    def init_random_model(self):
        pass

    def init_pretrained_model(self):
        pass

    def output_score(self, query_string, 
                    candidate_list, 
                    cut_tool=None):
        score_list = []
        for candidate in candidate_list:
            score = fuzz.ratio(" ".join(list(query_string)),
                                " ".join(list(candidate)))
            score_list.append(float(score)/float(100))
        print("---fuzzy distance-----", score_list)
        return [score_list]