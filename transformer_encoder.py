from model_template import ModelTemplate
import tensorflow as tf
import numpy as np
import transformer_utils
import rnn_utils, match_utils
from integration_func import generate_embedding_mat

class TransformerEncoder(ModelTemplate):
    def __init__(self):
        super(TransformerEncoder, self).__init__()

    def build_placeholder(self, config):
        
        self.config = config
        self.token_emb_mat = self.config["token_emb_mat"]
        self.vocab_size = int(self.config["vocab_size"])
        self.max_length = int(self.config["max_length"])
        self.emb_size = int(self.config["emb_size"])
        self.extra_symbol = self.config["extra_symbol"]
        self.scope = self.config["scope"]
        self.num_features = int(self.config["num_features"])
        self.num_classes = int(self.config["num_classes"])
        self.batch_size = int(self.config["batch_size"])
        self.grad_clipper = float(self.config.get("grad_clipper", 10.0))
        self.ema = self.config.get("ema", True)
        
        print("--------vocab size---------", self.vocab_size)
        print("--------max length---------", self.max_length)
        print("--------emb size-----------", self.emb_size)
        print("--------extra symbol-------", self.extra_symbol)
        print("--------emb matrix---------", self.token_emb_mat.shape)
        
        # ---- place holder -----
        self.sent1_token = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='sent1_token')
        self.sent2_token = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='sent2_token')
        self.gold_label = tf.placeholder(tf.int32, [self.batch_size], name='gold_label')
        self.sent1_len = tf.placeholder(tf.int32, [self.batch_size], name='sent1_token_length')
        self.sent2_len = tf.placeholder(tf.int32, [self.batch_size], name='sent2_token_length')
        
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        
        self.features = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_features], name="features")
        
        self.one_hot_label = tf.one_hot(self.gold_label, 2)
        
        # ------------ other ---------
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
        
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
        self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask, tf.int32), -1)
        
        # ---------------- for dynamic learning rate -------------------
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        self.learning_rate_value = float(self.config["learning_rate"])
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.emb_mat = generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                 init_mat=self.token_emb_mat, 
                                 extra_symbol=self.extra_symbol, 
                                 scope='gene_token_emb_mat')
        
        self.s1_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent1_token)  # bs,sl1,tel
        self.s2_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent2_token)  # bs,sl2,tel
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate_updated = False
        # ------ start ------
        self.pred_probs = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.opt = None
        self.train_op = None 

    def build_model(self):
        self.build_op()

    def _build_network(self, sent, emb, emb_mask, emb_len):
        position_encoding = self.config.get("position_encoding", "sinusoid")
        with tf.variable_scope("transformer_encoder") as scope:
            if position_encoding == "sinusoid":
                emb += transformer_utils.positional_encoding(sent,
                                        num_units=self.hidden_units, 
                                        zero_pad=False, 
                                        scale=False,
                                        scope="enc_pe")

            else:
                emb += transformer_utils.embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(sent)[1]), 0), [tf.shape(sent)[0], 1]),
                                    vocab_size=emb_len, 
                                    num_units=self.hidden_units, 
                                    zero_pad=False, 
                                    scale=False,
                                    scope="enc_pe")

            enc = tf.nn.dropout(emb, self.dropout_keep_prob)
            ## Blocks
            num_blocks = self.config.get("num_blocks", 6)
            num_heads = self.config.get("num_heads", 8)
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = transformer_utils.multihead_attention(queries=enc, 
                                            keys=enc,
                                            scope="multihead_attention_{}".format(i),
                                            num_units=self.hidden_units, 
                                            num_heads=num_heads, 
                                            dropout_rate=self.dropout_keep_prob,
                                            is_training=self.is_train,
                                            causality=False)
                    
                    ### Feed Forward
                    enc = transformer_utils.feedforward(enc, 
                                    num_units=[4*self.hidden_units, 
                                                self.hidden_units],
                                    scope="ffn_{}".format(i))

            enc = rnn_utils.task_specific_attention(enc, 
                                                self.hidden_units, 
                                                emb_mask)

            return enc

    def build_network(self):
        self.hidden_units = self.config["hidden_units"]
        with tf.variable_scope(self.scope+"-transformer-encoder"):
            self.feature_one = self._build_network(self.sent1_token,
                                            self.s1_emb, 
                                            self.sent1_token_mask, 
                                            self.sent1_token_len)

            tf.get_variable_scope().reuse_variables()
            self.feature_two = self._build_network(self.sent2_token,
                                            self.s2_emb, 
                                            self.sent2_token_mask, 
                                            self.sent2_token_len)

            self.output_features = tf.concat([self.feature_one, self.feature_two, 
                             self.feature_one-self.feature_two, 
                             self.feature_one*self.feature_two], axis=-1)

            match_utils.add_reg_without_bias(self.scope)

        if self.config["loss_type"] == "cross_entropy" or self.config["loss_type"] == "focal_loss":
            with tf.variable_scope(self.scope+"_prediction_layer"): 
                self.estimation = tf.layers.dense(
                    self.output_features,
                    units=self.num_classes,
                    name="prediction_layer")
            
                self.pred_probs = tf.contrib.layers.softmax(self.estimation)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
            
        elif self.config["loss_type"] == "contrastive_loss":
            if self.config["distance_metric"] == "l1_similarity":
                self.pred_probs = rnn_utils.l1_similarity(self.feature_one, self.feature_two)
                self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)
                    
            elif self.config["distance_metric"] == "l2_similarity":
                self.distance, self.contrastive_distance = rnn_utils.l2_similarity(self.feature_one, self.feature_two, self.one_hot_label)
                probs = tf.expand_dims(self.distance, -1)
                self.pred_probs = tf.concat([probs, (1.0 - probs)], axis=-1)
                # less than 0.5 is set to positive
                self.logits = tf.cast(tf.cast(tf.maximum((0.5 - self.distance), 0), tf.bool), tf.int32) 
        
            elif self.config["distance_metric"] == "cosine_contrastive":
                distance, self.contrastive_distance = rnn_utils.cosine_constrastive(self.feature_one, self.feature_two, self.one_hot_label)
                self.distance = (distance + 1) / 2
                probs = tf.expand_dims(self.distance, -1)
                self.pred_probs = tf.concat([1.0 - probs, probs], axis=-1)
                # larger than 0.5 is set to positive
                self.logits = tf.cast(tf.cast(tf.maximum((self.distance - 0.5), 0), tf.bool), tf.int32)
 
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

    def build_loss(self):
        self.wd = self.config.get("l2_reg", None)

        with tf.name_scope("loss"):
            if self.config["loss_type"] == "cross_entropy":
                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.estimation, 
                        labels=self.gold_label))
                    
            elif self.config["loss_type"] == "focal_loss":
                one_hot_target = tf.one_hot(self.gold_label, 2)
                print(one_hot_target.get_shape())
                self.loss = focal_loss(self.estimation, one_hot_target)
                
            elif self.config["loss_type"] == "contrastive_loss":
                if self.config["distance_metric"] == "l1_similarity":
                    self.loss = tf.reduce_mean(
                        tf.losses.softmax_cross_entropy(self.one_hot_label, 
                                                        self.pred_probs))
                    
                elif self.config["distance_metric"] == "cosine_constrastive":
                    self.loss = tf.reduce_mean(self.contrastive_distance)
                    
                elif self.config["distance_metric"] == "l2_similarity":
                    self.loss = tf.reduce_mean(self.contrastive_distance)
                   
            wd_loss = 0
            print("------------wd--------------", self.wd)
            if self.wd is not None:
                for var in set(tf.get_collection('reg_vars', self.scope)):
                    weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                wd_loss += weight_decay
                print("---------using l2 regualarization------------")
            self.loss += wd_loss

        tf.add_to_collection('ema/scalar', self.loss)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
 