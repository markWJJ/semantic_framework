from model_template import ModelTemplate
import tensorflow  as tf
import numpy as np
import rnn_utils

class QACNN(ModelTemplate):
    def __init__(self):
        super(QACNN, self).__init__()

    def build_model(self):
        self.build_op()

    # Hidden Layer
    def add_hl(self, q_embed, aplus_embed):
        with tf.variable_scope('HL'):
            W = tf.get_variable('weights', shape=[self.emb_size, self.hidden_size], initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.hidden_size]))
            h_q = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(q_embed, [-1, self.emb_size]), W)+b), [-1, self.max_length, self.hidden_size])
            h_ap = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aplus_embed, [-1, self.emb_size]), W)+b), [-1, self.max_length, self.hidden_size])
            return h_q, h_ap

     # CNN layer
    def add_model(self, h_q, h_ap):
        pool_q = list()
        pool_ap = list()
        h_q = tf.reshape(h_q, [-1, self.max_length, self.hidden_size, 1])
        h_ap = tf.reshape(h_ap, [-1, self.max_length, self.hidden_size, 1])
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                # filter W and b
                conv1_W = tf.get_variable('W', shape=[filter_size, self.hidden_size, 1, self.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv1_b = tf.get_variable('conv_b', initializer=tf.constant(0.1, shape=[self.num_filters]))
                # pooling bias,Q and A
                pool_qb = tf.get_variable('pool_qb', initializer=tf.constant(0.1, shape=[self.num_filters]))
                pool_ab = tf.get_variable('pool_ab', initializer=tf.constant(0.1, shape=[self.num_filters]))
                # conv
                out_q = tf.nn.relu((tf.nn.conv2d(h_q, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                # pool
                out_q = tf.nn.max_pool(out_q, [1,self.max_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_q = tf.nn.tanh(out_q+pool_qb)
                pool_q.append(out_q)

                out_ap = tf.nn.relu((tf.nn.conv2d(h_ap, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                out_ap = tf.nn.max_pool(out_ap, [1,self.max_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_ap = tf.nn.tanh(out_ap+pool_ab)
                pool_ap.append(out_ap)

        total_channels = len(self.filter_sizes)*self.num_filters

        real_pool_q = tf.reshape(tf.concat(pool_q, 3), [-1, total_channels])
        real_pool_ap = tf.reshape(tf.concat(pool_ap, 3), [-1, total_channels])

        return real_pool_q, real_pool_ap

    def build_network(self):
        self.wd = (self.config.get("l2_reg", None))
        self.filter_sizes = self.config.get("filter_sizes", [1,2,3,5])
        self.num_filters = self.config.get("num_filters", 512)
        self.margin = float(self.config.get("margin", 0.05))
        self.hidden_size = self.config.get("hidden_units", 100)
        with tf.variable_scope(self.scope+"-qacnn"):
            [self.h_q, self.h_ap] = self.add_hl(self.s1_emb, self.s2_emb)
            real_pool_q, real_pool_ap = self.add_model(self.h_q, self.h_ap)
            self.feature_one = real_pool_q
            self.feature_two = real_pool_ap

            self.output_features = tf.concat([self.feature_one, self.feature_two, 
                             self.feature_one-self.feature_two, 
                             self.feature_one*self.feature_two], axis=-1)

            rnn_utils.add_reg_without_bias(self.scope)

    def build_loss(self):

        with tf.name_scope("loss"):
            wd_loss = 0
            if self.wd is not None:
                for var in set(tf.get_collection('reg_vars', self.scope)):
                    weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                wd_loss += weight_decay

            if self.config["distance_metric"] == "l2_similarity":
                distance, contrastive_distance = rnn_utils.l2_similarity(self.feature_one, self.feature_two, self.one_hot_label)
                probs = tf.expand_dims(distance, -1)
                self.pred_probs = tf.concat([probs, (1.0 - probs)], axis=-1)
                # less than 0.5 is set to positive
                self.logits = tf.cast(tf.cast(tf.maximum((0.5 - distance), 0), tf.bool), tf.int32) 
            
            elif self.config["distance_metric"] == "cosine_contrastive":
                distance, contrastive_distance = rnn_utils.cosine_constrastive(self.feature_one, self.feature_two, self.one_hot_label)
                probs = tf.expand_dims(distance, -1)
                self.pred_probs = tf.concat([1.0 - probs, probs], axis=-1)
                # larger than 0.5 is set to positive
                self.logits = tf.cast(tf.cast(tf.maximum((distance - 0.5), 0), tf.bool), tf.int32)

            self.loss = tf.add(tf.reduce_mean(contrastive_distance), wd_loss, name="loss")
                   
        tf.add_to_collection('ema/scalar', self.loss)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name) 

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
            }
        elif data_type == "infer":
            [sent1_token_b, sent2_token_b, _, 
             sent1_token_len, sent2_token_len] = sample_batch

            feed_dict = {
                self.sent1_token: sent1_token_b,
                self.sent2_token: sent2_token_b,
                self.is_train: False
            }
        return feed_dict

    def build_accuracy(self):
        correct = tf.equal(
            self.logits,
            self.gold_label
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))    



