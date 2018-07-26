from model_template import ModelTemplate
import numpy as np
import tensorflow as tf
from disan_utils import disan, linear 

class DiSAN(ModelTemplate):
	def __init__(self):
		super(DiSAN, self).__init__()

	def build_model(self):
		self.build_op()

	def build_network(self):
		self.ocd = self.config["out_channel_dims"]
		self.fh = self.config["filter_heights"]
		self.hn = self.config["hidden_units"]
		self.wd = self.config["weight_decay"]
		self.dropout_keep_prob_float = self.config["dropout_keep_prob"]

		with tf.variable_scope(self.scope+'_sent_enc'):
			self.s1_rep = disan(
                self.s1_emb, self.sent1_token_mask, self.scope+'_DiSAN', 
                self.dropout_keep_prob_float, self.is_train, self.wd,
                'elu', None, 's1'
            )

			tf.get_variable_scope().reuse_variables()

			self.s2_rep = disan(
                self.s2_emb, self.sent2_token_mask, self.scope+'_DiSAN', 
                self.dropout_keep_prob_float, self.is_train, self.wd,
                'elu', None, 's2'
            )

		with tf.variable_scope(self.scope+'_output'):
			self.out_rep = tf.concat([self.s1_rep, self.s2_rep, 
            					self.s1_rep - self.s2_rep, 
            					self.s1_rep * self.s2_rep], -1)

			self.pre_output = tf.nn.elu(linear([self.out_rep], self.hn, True, 0., scope= self.scope+'_pre_output', squeeze=False,
                                           wd=self.wd, input_keep_prob=self.dropout_keep_prob_float,
                                           is_train=self.is_train))
			self.output_features = self.pre_output
			self.estimation = linear([self.pre_output], self.num_classes, True, 0., scope= self.scope+'_pre_softmax', squeeze=False,
                            wd=self.wd, input_keep_prob=self.dropout_keep_prob_float,
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
                self.learning_rate: self.learning_rate_value
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
    





