import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
import layer_utils
import match_utils
from model_template import ModelTemplate
from loss_utils import focal_loss

	


class BIMPM_NEW(ModelTemplate):
	def __init__(self):
		super(BIMPM_NEW, self).__init__()

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
		self.options = self.config["options"]
		self.options["batch_size"] = self.batch_size
		self.highway_layer_num = self.options["highway_layer_num"]
		self.with_highway = self.options["with_highway"]
		self.wd = self.config.get("weight_decay", None)
		self.l2_reg = float(self.config["l2_reg"])

		in_question_repres = tf.nn.dropout(self.s1_emb, self.dropout_keep_prob)
		in_passage_repres = tf.nn.dropout(self.s2_emb, self.dropout_keep_prob)

		input_dim = self.emb_size

		# ======Highway layer======
		if self.with_highway:
			with tf.variable_scope(self.scope+"-input_highway"):
				in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, self.highway_layer_num)
				tf.get_variable_scope().reuse_variables()
				in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, self.highway_layer_num)

		# ========Bilateral Matching=====
		with tf.variable_scope(self.scope+"-bilateral_matching"): 
			(match_representation, match_dim) = match_utils.bilateral_match_func(
						in_question_repres, in_passage_repres,
						self.sent1_token_len, self.sent2_token_len, 
						self.sent1_token_mask, self.sent2_token_mask, input_dim, self.config["mode"], 
						options=self.options, dropout_rate=self.dropout_keep_prob)
			self.output_features = match_representation

		#========Prediction Layer=========
		with tf.variable_scope(self.scope+"-prediction"): 
			# match_dim = 4 * self.options.aggregation_lstm_dim
			w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
			b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
			w_1 = tf.get_variable("w_1", [match_dim/2, self.num_classes],dtype=tf.float32)
			b_1 = tf.get_variable("b_1", [self.num_classes],dtype=tf.float32)

			# if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
			logits = tf.matmul(match_representation, w_0) + b_0
			logits = tf.tanh(logits)
			logits = tf.nn.dropout(logits, (self.dropout_keep_prob))
			self.estimation = tf.matmul(logits, w_1) + b_1

			self.pred_probs = tf.contrib.layers.softmax(self.estimation)
			self.logits = tf.cast(tf.argmax(self.pred_probs, -1), tf.int32)

			match_utils.add_reg_without_bias(self.scope)

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





