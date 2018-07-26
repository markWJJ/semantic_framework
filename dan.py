import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope
import layer_utils
import match_utils
from model_template import ModelTemplate
from loss_utils import focal_loss
import rnn_utils, match_utils

class DAN(ModelTemplate):
	def __init__(self):
		super(DAN, self).__init__()

	def build_model(self):
		self.build_op()

	def _build_network(self, emb, emb_mask, emb_len):
		with tf.variable_scope("self-attention-dan") as scope:

			emb = tf.nn.dropout(emb, self.dropout_keep_prob)
			encoder_fw, encoder_bw = rnn_utils.BiRNN(emb, self.dropout_keep_prob, 
										"tied_encoder", 
										emb_len, 
										self.hidden_units, 
										self.is_train, self.rnn_cell)

			encoder_represnetation = tf.concat([encoder_fw, encoder_bw], axis=-1)
			emd_mask = tf.cast(emb_mask, tf.float32)
			representation = rnn_utils.task_specific_attention(encoder_represnetation, 
												self.hidden_units, 
												emb_mask)

			deep_self_att = rnn_utils.highway_net(representation, 
									self.config["num_layers"],
									self.dropout_keep_prob,
									batch_norm=False,
									training=self.is_train)
			return deep_self_att

	def build_network(self):
		self.hidden_units = self.config["hidden_units"]
		self.rnn_cell = self.config["rnn_cell"]
		with tf.variable_scope(self.scope+"-deep-averaging"):
			self.feature_one = self._build_network(self.s1_emb, 
											self.sent1_token_mask, 
											self.sent1_token_len)
			tf.get_variable_scope().reuse_variables()
			self.feature_two = self._build_network(self.s2_emb, 
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