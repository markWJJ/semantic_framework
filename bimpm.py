import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
from rnn_utils import (bilateral_matching, mean_pool, last_relevant_output, 
                        BiRNN, create_tied_encoder, add_reg_without_bias)
from model_template import ModelTemplate
from loss_utils import focal_loss

class BIMPM(ModelTemplate):
    def __init__(self):
        super(BIMPM, self).__init__()

    def build_model(self):
        self.build_op()
    
    def build_loss(self):

        with tf.name_scope("loss"):
            wd_loss = 0
            print("------------wd--------------", self.wd)
            if self.wd is not None:
                for var in set(tf.get_collection('reg_vars', self.scope)):
                    weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                wd_loss += weight_decay
                print("---------using l2 regualarization------------")

            if self.config["loss_type"] == "cross_entropy":
                self.loss = tf.add(
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, labels=self.gold_label)),
                    wd_loss, name="loss")
            elif self.config["loss_type"] == "focal_loss":
                one_hot_target = tf.one_hot(self.gold_label, 2)
                print(one_hot_target.get_shape())
                self.loss = focal_loss(self.estimation, one_hot_target) + wd_loss
                   
        tf.add_to_collection('ema/scalar', self.loss)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        
    def build_network(self):
        
        self.l2_reg = float(self.config["l2_reg"])
        self.aggregation_rnn_hidden_size = self.config["aggregation_rnn_hidden_size"]
        self.hidden_units = self.config["hidden_units"]
        self.rnn_cell = self.config["rnn_cell"]
        self.wd = self.config.get("weight_decay", None)
        
        [anchor_encoder_fw, 
         anchor_encoder_bw, 
         check_encoder_fw, 
         check_encoder_bw] = create_tied_encoder(self.s1_emb, self.s2_emb, self.sent1_token_len, self.sent2_token_len, 
                                    self.dropout_keep_prob, self.hidden_units, self.is_train, self.rnn_cell, self.scope)

        with tf.variable_scope(self.scope+"_matching_layer"):
            # Shapes: (batch_size, num_sentence_words, 8*multiperspective_dims)
            sent1_token_mask = tf.cast(self.sent1_token_mask, tf.int32)
            sent2_token_mask = tf.cast(self.sent2_token_mask, tf.int32)
            match_one_to_two_out, match_two_to_one_out = bilateral_matching(
                anchor_encoder_fw, anchor_encoder_bw,
                check_encoder_fw, check_encoder_bw,
                sent1_token_mask, sent2_token_mask, self.is_train,
                self.dropout_keep_prob)
            
        with tf.variable_scope(self.scope+"_aggregation_layer"):
            aggregated_representations = []
            sentence_one_aggregation_input = match_one_to_two_out
            sentence_two_aggregation_input = match_two_to_one_out

            with tf.variable_scope(self.scope+"_aggregate_sentence_one"):
                if self.rnn_cell == "lstm_cell":
                    aggregation_lstm_fw = tf.contrib.rnn.LSTMCell(self.aggregation_rnn_hidden_size)
                    aggregation_lstm_bw = tf.contrib.rnn.LSTMCell(self.aggregation_rnn_hidden_size)
                elif self.rnn_cell == "gru_cell":
                    aggregation_lstm_fw = tf.contrib.rnn.GRUCell(self.aggregation_rnn_hidden_size)
                    aggregation_lstm_bw = tf.contrib.rnn.GRUCell(self.aggregation_rnn_hidden_size)
                # Encode the matching output into a fixed size vector.
                # Shapes: (batch_size, num_sentence_words, aggregation_rnn_hidden_size)
                (fw_agg_outputs, bw_agg_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=aggregation_lstm_fw,
                    cell_bw=aggregation_lstm_bw,
                    dtype="float",
                    sequence_length=self.sent1_token_len,
                    inputs=sentence_one_aggregation_input,
                    scope="encode_sentence_one_matching_vectors")
                d_fw_agg_outputs = tf.nn.dropout(
                    fw_agg_outputs,
                    keep_prob=self.dropout_keep_prob,
                    name="sentence_one_fw_agg_outputs_dropout")
                d_bw_agg_outputs = tf.nn.dropout(
                    bw_agg_outputs,
                    keep_prob=self.dropout_keep_prob,
                    name="sentence_one_bw_agg_outputs_dropout")

                # Get the last output (wrt padding) of the LSTM.
                # Shapes: (batch_size, aggregation_rnn_hidden_size)
                last_fw_agg_output = last_relevant_output(d_fw_agg_outputs,
                                                          self.sent1_token_len)
                last_bw_agg_output = last_relevant_output(d_bw_agg_outputs,
                                                          self.sent1_token_len)
                aggregated_representations.append(last_fw_agg_output)
                aggregated_representations.append(last_bw_agg_output)

            with tf.variable_scope(self.scope+"_aggregate_sentence_two"):
                if self.rnn_cell == "lstm_cell":
                    aggregation_lstm_fw = tf.contrib.rnn.LSTMCell(self.aggregation_rnn_hidden_size)
                    aggregation_lstm_bw = tf.contrib.rnn.LSTMCell(self.aggregation_rnn_hidden_size)
                elif self.rnn_cell == "gru_cell":
                    aggregation_lstm_fw = tf.contrib.rnn.GRUCell(self.aggregation_rnn_hidden_size)
                    aggregation_lstm_bw = tf.contrib.rnn.GRUCell(self.aggregation_rnn_hidden_size)
                # Encode the matching output into a fixed size vector.
                # Shapes: (batch_size, num_sentence_words, aggregation_rnn_hidden_size)
                (fw_agg_outputs, bw_agg_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=aggregation_lstm_fw,
                    cell_bw=aggregation_lstm_bw,
                    dtype="float",
                    sequence_length=self.sent2_token_len,
                    inputs=sentence_two_aggregation_input,
                    scope="encode_sentence_two_matching_vectors")
                d_fw_agg_outputs = tf.nn.dropout(
                    fw_agg_outputs,
                    keep_prob=self.dropout_keep_prob,
                    name="sentence_two_fw_agg_outputs_dropout")
                d_bw_agg_outputs = tf.nn.dropout(
                    bw_agg_outputs,
                    keep_prob=self.dropout_keep_prob,
                    name="sentence_two_bw_agg_outputs_dropout")

                # Get the last output (wrt padding) of the LSTM.
                # Shapes: (batch_size, aggregation_rnn_hidden_size)
                last_fw_agg_output = last_relevant_output(d_fw_agg_outputs, self.sent2_token_len)
                last_bw_agg_output = last_relevant_output(d_bw_agg_outputs, self.sent2_token_len)
                
                aggregated_representations.append(last_fw_agg_output)
                aggregated_representations.append(last_bw_agg_output)
            # Combine the 4 aggregated representations (fw a to b, bw a to b,
            # fw b to a, bw b to a)
            # Shape: (batch_size, 4*aggregation_rnn_hidden_size)
            combined_aggregated_representation = tf.concat(aggregated_representations, 1)
            self.output_features = combined_aggregated_representation

        with tf.variable_scope(self.scope+"_prediction_layer"):
            # Now, we pass the combined aggregated representation
            # through a 2-layer feed forward NN.
            predict_layer_one_out = tf.layers.dense(
                combined_aggregated_representation,
                combined_aggregated_representation.get_shape().as_list()[1],
                activation=tf.nn.tanh,
                name="prediction_layer_one")
            d_predict_layer_one_out = tf.nn.dropout(
                predict_layer_one_out,
                keep_prob=self.dropout_keep_prob,
                name="prediction_layer_dropout")
            self.estimation = tf.layers.dense(
                d_predict_layer_one_out,
                units=2,
                name="prediction_layer_two")
            
            self.pred_probs = tf.contrib.layers.softmax(self.estimation)
            self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
            add_reg_without_bias(self.scope)

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