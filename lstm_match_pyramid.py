from model_template import ModelTemplate
import numpy as np
import tensorflow as tf
from loss_utils import focal_loss
import rnn_utils, match_utils

class LSTMMatchPyramid(ModelTemplate):
    def __init__(self):
        super(LSTMMatchPyramid, self).__init__()
         
    def build_model(self):
        self.build_op()
        
    def build_network(self):
        self.l2_reg = float(self.config["l2_reg"])
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index', 
                               shape=[None, self.config["max_length"], self.config["max_length"], 3])
        
        # batch_size * X1_maxlen * X2_maxlen

        self.hidden_units = self.config["hidden_units"]
        self.rnn_cell = self.config["rnn_cell"]

        s1_emb = tf.nn.dropout(self.s1_emb, self.dropout_keep_prob)
        s2_emb = tf.nn.dropout(self.s2_emb, self.dropout_keep_prob)

        [anchor_encoder_fw, 
         anchor_encoder_bw, 
         check_encoder_fw, 
         check_encoder_bw] = rnn_utils.create_tied_encoder(s1_emb, s2_emb, self.sent1_token_len, 
                                   self.sent2_token_len, self.dropout_keep_prob, self.hidden_units, 
                                   self.is_train, self.rnn_cell, self.scope+"_tied_lstm_encoder")
        
        anchor_encoder = tf.concat([anchor_encoder_fw, anchor_encoder_bw], 2)
        check_encoder = tf.concat([check_encoder_fw, check_encoder_bw], 2)

        self.cross = tf.einsum('abd,acd->abc', anchor_encoder, check_encoder)
        self.cross_img = tf.expand_dims(self.cross, 3)
        
        with tf.variable_scope(self.scope+'_conv_pooling_layer'):
        
            # convolution
            self.w1 = tf.get_variable('w1', 
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.2, dtype=tf.float32), 
                         dtype=tf.float32, 
                         shape=[2, 10, 1, 8])
        
            self.b1 = tf.get_variable('b1', initializer=tf.constant_initializer(0.0), dtype=tf.float32, shape=[8])
            # batch_size * X1_maxlen * X2_maxlen * feat_out
            self.conv1 = tf.nn.relu(tf.nn.conv2d(self.cross_img, self.w1, [1, 1, 1, 1], "SAME") + self.b1)
        
            # dynamic pooling
            self.conv1_expand = tf.gather_nd(self.conv1, self.dpool_index)
            self.pool1 = tf.nn.max_pool(self.conv1_expand, 
                        [1, self.config['max_length'] / self.config['sent1_psize'], 
                        self.config['max_length'] / self.config['sent2_psize'], 1], 
                        [1, self.config['max_length'] / self.config['sent1_psize'], 
                       self.config['max_length'] / self.config['sent2_psize'], 1], "VALID")
                                  
        with tf.variable_scope(self.scope+'_feature_layer'):
            flatten = tf.reshape(self.pool1, 
                          [-1, self.config['sent1_psize'] * self.config['sent2_psize'] * 8])
            print(flatten.get_shape())

            self.output_features = rnn_utils.highway_net(flatten, 
                                    self.config["num_layers"],
                                    self.dropout_keep_prob,
                                    batch_norm=False,
                                    training=self.is_train)
        
        with tf.variable_scope(self.scope+"_output_layer"):
        
            self.estimation = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=self.num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC")       

        match_utils.add_reg_without_bias(self.scope)
        self.pred_probs = tf.contrib.layers.softmax(self.estimation)
        self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
         
    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)
                                  
    def get_feed_dict(self, sample_batch, dropout_keep_prob, data_type='train'):
        if data_type == "train" or data_type == "test":
            [sent1_token_b, sent2_token_b, gold_label_b, 
             sent1_token_len, sent2_token_len] = sample_batch

            feed_dict = {
                self.sent1_token: sent1_token_b,
                self.sent2_token: sent2_token_b,
                self.gold_label: gold_label_b,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0,
                self.is_train: True if data_type == 'train' else False,
                self.dpool_index: self.dynamic_pooling_index(sent1_token_len, sent2_token_len, 
                                            self.config['max_length'], self.config['max_length']),
                self.learning_rate: self.learning_rate_value}
        elif data_type == "infer":
            [sent1_token_b, sent2_token_b, _, 
             sent1_token_len, sent2_token_len] = sample_batch

            feed_dict = {
                self.sent1_token: sent1_token_b,
                self.sent2_token: sent2_token_b,
                self.is_train: False,
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0,
                self.dpool_index: self.dynamic_pooling_index(sent1_token_len, sent2_token_len, 
                                            self.config['max_length'], self.config['max_length'])}
        return feed_dict

    def build_loss(self):
        self.wd = self.config["weight_decay"]

        self.loss = 0
        for var in set(tf.get_collection('reg_vars', self.scope)):
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            self.loss += weight_decay

        with tf.name_scope("loss"):
            if self.config["loss_type"] == "cross_entropy":
                self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.gold_label))
            elif self.config["loss_type"] == "focal_loss":
                one_hot_target = tf.one_hot(self.gold_label, 2)
                print(one_hot_target.get_shape())
                self.loss += focal_loss(self.estimation, one_hot_target)
        
        tf.add_to_collection('ema/scalar', self.loss)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)

    def build_accuracy(self):
        correct = tf.equal(
            self.logits,
            self.gold_label
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))    