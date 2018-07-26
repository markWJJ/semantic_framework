from model_template import ModelTemplate
import tensorflow as tf

import biblosa_utils

class BiBLOSA(ModelTemplate):
    def __init__(self):
        super(BiBLOSA, self).__init__()

    def build_model(self):
        self.build_op()

    def build_network(self):

        self.tds = self.vocab_size
        self.wd = self.config.get("weight_decay", 1e-5)

        self.tensor_dict = {}

        self.dropout_keep_prob_float = self.config.get("dropout_keep_prob", 0.80)
        
         # ----------- parameters -------------
        self.tl = self.max_length
        self.tel = self.emb_size

        self.ocd = self.config["out_channel_dims"]
        self.fh = self.config["filter_heights"]
        self.hn = self.config["hidden_units"]

        self.output_class = self.num_classes
        self.bs = tf.shape(self.sent1_token)[0]
        self.sl1 = tf.shape(self.sent1_token)[1]
        self.sl2 = tf.shape(self.sent2_token)[1]

        in_question_repres = tf.nn.dropout(self.s1_emb, self.dropout_keep_prob)
        in_passage_repres = tf.nn.dropout(self.s2_emb, self.dropout_keep_prob)

        with tf.variable_scope(self.scope+'-sent_encoding'):
            act_func_str = 'elu' if self.config["context_fusion_method"] in ['block', 'disa'] else 'relu'

            s1_rep = biblosa_utils.sentence_encoding_models(
                in_question_repres, self.sent1_token_mask, 
                self.config["context_fusion_method"], 
                act_func_str,
                'ct_based_sent2vec', 
                self.wd,
                self.is_train, 
                self.dropout_keep_prob_float,
                block_len=self.config["block_len"], 
                hn=self.hn)

            tf.get_variable_scope().reuse_variables()

            s2_rep = biblosa_utils.sentence_encoding_models(
                in_passage_repres, self.sent2_token_mask, 
                self.config["context_fusion_method"], 
                act_func_str,
                'ct_based_sent2vec', 
                self.wd, 
                self.is_train, 
                self.dropout_keep_prob_float,
                block_len=self.config["block_len"], 
                hn=self.hn)

            self.tensor_dict['s1_rep'] = s1_rep
            self.tensor_dict['s2_rep'] = s2_rep

        with tf.variable_scope(self.scope+'-output'):
            act_func = tf.nn.elu if self.config["context_fusion_method"] in ['block', 'disa'] else tf.nn.relu

            self.out_rep = tf.concat([s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)
            pre_output = act_func(biblosa_utils.linear([self.out_rep], self.hn, True, 0., 
                                            scope= 'pre_output', squeeze=False,
                                            wd=self.wd, 
                                            input_keep_prob=self.dropout_keep_prob_float,
                                            is_train=self.is_train))
            self.pre_output = biblosa_utils.highway_net(
                pre_output, self.hn, True, 0., 'pre_output1', 
                act_func_str, False, self.wd,
                self.dropout_keep_prob_float, self.is_train)

            self.output_features = self.pre_output

            self.estimation = biblosa_utils.linear([self.output_features], 
                            self.num_classes, True, 0., 
                            scope= 'logits', squeeze=False,
                            wd=self.wd, 
                            input_keep_prob=self.dropout_keep_prob_float,
                            is_train=self.is_train)
            self.pred_probs = tf.contrib.layers.softmax(self.estimation)
            self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
            print("------------Succeeded in building network-------------")


    def build_loss(self):
        self.loss = 0
        for var in set(tf.get_collection('reg_vars', self.scope)):
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            self.loss += weight_decay
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=self.estimation
        )

        print("------------Succeeded in building loss function------------")
        self.loss += tf.reduce_mean(losses, name='xentropy_loss_mean')
        tf.add_to_collection('ema/scalar', self.loss)


    def get_feed_dict(self, sample_batch, dropout_keep_prob, data_type='train'):
        if data_type == "train" or data_type == "test":
            [sent1_token_b, sent2_token_b, gold_label_b, 
            sent1_token_len, sent2_token_len] = sample_batch

            feed_dict = {
                self.sent1_token: sent1_token_b,
                self.sent2_token: sent2_token_b,
                self.gold_label:gold_label_b,
                self.is_train: True if data_type == 'train' else False,
                self.learning_rate: self.learning_rate_value,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 0.9
            }
        elif data_type == "infer":
            [sent1_token_b, sent2_token_b, _, 
            sent1_token_len, sent2_token_len] = sample_batch

            feed_dict = {
                self.sent1_token: sent1_token_b,
                self.sent2_token: sent2_token_b,
                self.is_train: False,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0
            }
        return feed_dict

    def build_accuracy(self):
        correct = tf.equal(
            self.logits,
            self.gold_label
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    

 

          



