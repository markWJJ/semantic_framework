import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from rnn_utils import (bilateral_matching, mean_pool, last_relevant_output, 
               BiRNN, create_tied_encoder, l1_similarity, cosine_constrastive, l2_similarity)
from model_template import ModelTemplate
from loss_utils import focal_loss

class SiameseLSTM(ModelTemplate):
    def __init__(self):
        super(SiameseLSTM, self).__init__()
        
    def build_model(self):
        self.build_op()
        
    def build_loss(self):

        with tf.name_scope("loss"):
            if self.config["loss_type"] == "cross_entropy":
                self.loss = tf.add(
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.gold_label)),
                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
                
            elif self.config["loss_type"] == "focal_loss":
                one_hot_target = tf.one_hot(self.gold_label, 2)
                print(one_hot_target.get_shape())
                self.loss = focal_loss(self.estimation, one_hot_target)
                
            elif self.config["loss_type"] == "contrastive_loss":
                if self.config["distance_metric"] == "l1_similarity":
                    self.loss = tf.add(
                    tf.reduce_mean(tf.losses.softmax_cross_entropy(self.one_hot_label, self.pred_probs)),
                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
                    
                elif self.config["distance_metric"] == "cosine_constrastive":
                    self.loss = tf.add(
                    tf.reduce_mean(self.contrastive_distance),
                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
                    
                elif self.config["distance_metric"] == "l2_similarity":
                    self.loss = tf.add(
                    tf.reduce_mean(self.contrastive_distance),
                    tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name="loss")
                   
        tf.add_to_collection('ema/scalar', self.loss)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
            
    def build_network(self):
        
        self.hidden_units = self.config["hidden_units"]
        self.rnn_cell = self.config["rnn_cell"]

        s1_emb = tf.nn.dropout(self.s1_emb, self.dropout_keep_prob)
        s2_emb = tf.nn.dropout(self.s2_emb, self.dropout_keep_prob)

        [anchor_encoder_fw, 
         anchor_encoder_bw, 
         check_encoder_fw, 
         check_encoder_bw] = create_tied_encoder(s1_emb, s2_emb, self.sent1_token_len, 
                                   self.sent2_token_len, self.dropout_keep_prob, self.hidden_units, 
                                   self.is_train, self.rnn_cell, self.scope+"_tied_lstm_encoder")
        
        anchor_encoder = tf.concat([anchor_encoder_fw, anchor_encoder_bw], 2)
        check_encoder = tf.concat([check_encoder_fw, check_encoder_bw], 2)
        
        if self.config["feature_type"] == "mean_pool":
            pooled_fw_output_one = mean_pool(anchor_encoder_fw, self.sent1_token_len)
            pooled_bw_output_one = mean_pool(anchor_encoder_bw, self.sent1_token_len)
            
            pooled_fw_output_two = mean_pool(check_encoder_fw, self.sent2_token_len)
            pooled_bw_output_two = mean_pool(check_encoder_bw, self.sent2_token_len)
            # Shape: (batch_size, 2*rnn_hidden_size)
            self.feature_one = tf.concat([pooled_fw_output_one, pooled_bw_output_one], 1)
            self.feature_two = tf.concat([pooled_fw_output_two, pooled_bw_output_two], 1)
            
        elif self.config["feature_type"] == "last_pool":
            self.feature_one = last_relevant_output(anchor_encoder, self.sent1_token_len)
            self.feature_two = last_relevant_output(check_encoder, self.sent2_token_len)
            print("---------------succeeded in implementing last_pool------------")
                
        self.output_features = tf.concat([self.feature_one, self.feature_two, 
                             self.feature_one-self.feature_two, 
                             self.feature_one*self.feature_two], axis=-1)

        if self.config["loss_type"] == "cross_entropy" or self.config["loss_type"] == "focal_loss":
            with tf.variable_scope(self.scope+"_prediction_layer"): 
                self.estimation = tf.layers.dense(
                    self.output_features,
                    units=2,
                    name="prediction_layer")
                
                self.pred_probs = tf.contrib.layers.softmax(self.estimation)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
                
        elif self.config["loss_type"] == "contrastive_loss":
            if self.config["distance_metric"] == "l1_similarity":
                self.pred_probs = l1_similarity(self.feature_one, self.feature_two)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
                        
            elif self.config["distance_metric"] == "l2_similarity":
                self.distance, self.contrastive_distance = l2_similarity(self.feature_one, self.feature_two, self.one_hot_label)
                probs = tf.expand_dims(self.distance, -1)
                self.pred_probs = tf.concat([probs, (1.0 - probs)], axis=-1)
                # less than 0.5 is set to positive
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
            
            elif self.config["distance_metric"] == "cosine_contrastive":
                self.distance, self.contrastive_distance = cosine_constrastive(self.feature_one, self.feature_two, self.one_hot_label)
                probs = tf.expand_dims(self.distance, -1)
                self.pred_probs = tf.concat([1.0 - probs, probs], axis=-1)
                # larger than 0.5 is set to positive
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)

    def build_accuracy(self):
        correct = tf.equal(
            self.logits,
            self.gold_label
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
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
                self.dropout_keep_prob: dropout_keep_prob if data_type == 'train' else 1.0
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