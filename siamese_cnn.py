from model_template import ModelTemplate
import tensorflow as tf
import numpy as np
from cnn_utils import *
from loss_utils import focal_loss

class SiameseCNN(ModelTemplate):
    def __init__(self):
        super(SiameseCNN, self).__init__()
    
    def build_model(self):
        self.build_op()

    def build_network(self):
        
        """
        :param w: filter width
        :param l2_reg: L2 regularization coefficient
        :param model_type: Type of the network(BCNN, ABCNN1, ABCNN2, ABCNN3).
        :param d0: dimensionality of word embedding(default: 300)
        :param di: The number of convolution kernels (default: 50)
        :param num_layers: The number of convolution layers.
        """
        self.w = int(self.config["filter_width"])
        self.l2_reg = float(self.config["l2_reg"])
        self.model_type = self.config["model_type"]
        self.d0 = self.emb_mat.get_shape()[1] 
        self.di = int(self.config["di"])
        self.num_layers = int(self.config["num_layers"])
        
        transpose_s1_emb = tf.transpose(self.s1_emb, [0,2,1]) # batch_size x sequence_length x words_dimensionality
        transpose_s2_emb = tf.transpose(self.s2_emb, [0,2,1])
        
        x1_expanded = tf.expand_dims(transpose_s1_emb, -1)
        x2_expanded = tf.expand_dims(transpose_s2_emb, -1)

        LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope=self.scope+"CNN-0", x1=x1_expanded, x2=x2_expanded, 
                                s=self.max_length, d=self.d0, w=self.w, di=self.di, 
                                model_type=self.model_type, l2_reg=self.l2_reg)
        
        if self.num_layers > 1:
            LI_2, LO_2, RI_2, RO_2 = CNN_layer(variable_scope=self.scope+"CNN-1", x1=LI_1, x2=RI_1, s=self.max_length, 
                               d=self.di, w=self.w, di=self.di, model_type=self.model_type, l2_reg=self.l2_reg)
            h1 = LO_2
            h2 = RO_2
            print("------second cnn layer---------")
        else:
            h1 = LO_1
            h2 = RO_1
   
        with tf.variable_scope("feature_layer"):
        
            print(h1.get_shape())
        
            self.output_features = tf.concat([h1, h2, h1-h2, h1*h2], axis=1, name="output_features")
            
        with tf.variable_scope("output_layer"):

            self.estimation = tf.contrib.layers.fully_connected(
                inputs=self.output_features,
                num_outputs=self.num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.pred_probs = tf.contrib.layers.softmax(self.estimation)
        self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
        
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