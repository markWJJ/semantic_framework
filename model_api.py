import tensorflow as tf
import numpy as np
import pickle as pkl
import os, json, codecs, sys

class ModelAPI(object):
    def __init__(self, model_dir, embed_path):
        self.model_dir = model_dir
        self.embed_path = embed_path
        self.model_type = None
        
    def dump_config(self):
        pkl.dump(self.parameter_config, open(os.path.join(self.model_dir, "config.pkl"), "wb"))
        
    def load_config(self):
        self.parameter_config = json.load(open(os.path.join(self.model_dir, "config.json"), "r"))
        
        if sys.version_info < (3, ):
            embedding_info = pkl.load(open(os.path.join(self.embed_path, "emb_mat.pkl"), "rb"))
        else:
            embedding_info = pkl.load(open(os.path.join(self.embed_path, "emb_mat.pkl"), "rb"), encoding="iso-8859-1")
        self.config = {}
        self.config["token2id"] = embedding_info["token2id"]
        self.config["id2token"] = embedding_info["id2token"]
        self.config["token_emb_mat"] = embedding_info["embedding_matrix"]
        for key in self.parameter_config:
            self.config[key] = self.parameter_config[key]
        self.batch_size = self.config["batch_size"]

    def update_config(self, updated_config_dict):
        for key in updated_config_dict:
            if key in self.config:
                self.config[key] = updated_config_dict[key]
            else:
                self.config[key] = updated_config_dict[key]

                
    def iter_batch(self, anchor, check, label, anchor_len, check_len, batch_size, mode="train"):
        assert anchor.shape == check.shape
        if mode == "train":
            shuffled_index = np.random.permutation(anchor.shape[0])
        else:
            shuffled_index = range(anchor.shape[0])
        batch_num = int(anchor.shape[0] / batch_size)
        for t in range(batch_num):
            start_index = t * batch_size
            end_index = start_index + batch_size
            sub_anchor = anchor[shuffled_index[start_index:end_index]]
            sub_check = check[shuffled_index[start_index:end_index]]
            sub_label = label[shuffled_index[start_index:end_index]]
            sub_anchor_len = anchor_len[shuffled_index[start_index:end_index]]
            sub_check_len = check_len[shuffled_index[start_index:end_index]]
            yield sub_anchor, sub_check, sub_label, sub_anchor_len, sub_check_len
        if self.config["scope"] != "transformer_encoder" and self.config["scope"] != "dan_fast":
            if end_index < anchor.shape[0]:
                sub_anchor = anchor[shuffled_index[end_index:]]
                sub_check = check[shuffled_index[end_index:]]
                sub_label = label[shuffled_index[end_index:]]
                sub_anchor_len = anchor_len[shuffled_index[end_index:]]
                sub_check_len = check_len[shuffled_index[end_index:]]
                yield sub_anchor, sub_check, sub_label, sub_anchor_len, sub_check_len
        
    def build_graph(self, model, device="/cpu:0"):
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            g.device(device)
            session_conf = tf.ConfigProto(
              intra_op_parallelism_threads=20, # control inner op parallelism threads
              inter_op_parallelism_threads=20, # controal among op parallelism threads
              device_count={'CPU': 4, 'GPU': 1},
              allow_soft_placement=True,
              log_device_placement=False)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config["gpu_ratio"]) 
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.model = model
            self.model.build_placeholder(self.config)
            print("-------Succeeded in building placeholder---------")
            self.model.build_model()
            print("-------Succeeded in building model-------")
            self.model.init_step(self.sess)
            print("-------Succeeded in initializing model-------")
     
    def load_model(self, load_type):
        print(self.config["model_id"])
        self.model_id = self.config.get("model_id", None)
        if load_type == "specific" and self.model_id:
            model_path = os.path.join(self.model_dir, self.model_type+".ckpt-"+str(self.model_id))
            self.model.saver.restore(self.sess, model_path)
            print("-------------succeeded in restoring pretrinaiend model with model id------", self.model_id)
        elif load_type == "latest":
            print(tf.train.latest_checkpoint(self.model_dir))
            self.model.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
            print("-----------succeeded in restoring pretrained model------------")
                
    
    def infer_features(self, sample_batch):
        features = self.model.infer_features(self.sess, sample_batch, 
                                self.config["dropout_keep_prob"],"infer")
        return features

    def infer_step(self, sample_batch):
        [logits, pred_probs] = self.model.infer(self.sess, sample_batch, self.config["dropout_keep_prob"], "infer")
        return logits, pred_probs
           
    def train_step(self, train_dataset, dev_dataset):
        self.best_dev_accuracy = -100.0
        self.early_stop_step = self.config["early_stop_step"]
        [train_anchor, train_check, train_label, 
                               train_anchor_len, train_check_len] = train_dataset
        [dev_anchor, dev_check, dev_label, 
                               dev_anchor_len, dev_check_len] = dev_dataset

        train_cnt = 0
        stop_step = 0
        stop_flag = False
        for epoch in range(self.config["max_epoch"]):
            train_batch = self.iter_batch(train_anchor, train_check, train_label, 
                               train_anchor_len, train_check_len, self.batch_size)
            
            train_loss = 0.0
            train_accuracy = 0.0
            train_internal_cnt = 0
            for train in train_batch:
                [sub_anchor, sub_check, 
                sub_label, 
                sub_anchor_len, sub_check_len] = train
                
                
                [loss, train_op, 
                global_step, accuracy, preds] = self.model.step(self.sess, [sub_anchor, sub_check, sub_label, 
                                                      sub_anchor_len, sub_check_len], self.config["dropout_keep_prob"])
                
                train_cnt += 1
                train_internal_cnt += 1
                train_loss += loss
                train_accuracy += accuracy
                
                if np.mod(train_cnt, self.config["validation_step"]) == 0:
                    
                    dev_batch = self.iter_batch(dev_anchor, dev_check, dev_label, 
                               dev_anchor_len, dev_check_len, self.batch_size, "dev")
                    dev_cnt = 0
                    dev_accuracy = 0.0
                    dev_loss = 0.0
                    for dev in dev_batch:
                        [sub_anchor, sub_check, 
                        sub_label, 
                        sub_anchor_len, sub_check_len] = dev

                        [loss, logits, pred_probs, accuracy] = self.model.infer(self.sess, [sub_anchor, sub_check, sub_label, 
                                                  sub_anchor_len, sub_check_len], self.config["dropout_keep_prob"], "test")
                        dev_cnt += 1
                        dev_accuracy += accuracy
                        dev_loss += loss

                    dev_accuracy /= float(dev_cnt)
                    dev_loss /= float(dev_cnt)
                    if dev_accuracy > self.best_dev_accuracy:
                        self.best_dev_accuracy = dev_accuracy
                        self.model.saver.save(self.sess, 
                                       os.path.join(self.model_dir, self.config["model_type"]+".ckpt"), 
                                       global_step=global_step)
                        stop_step = 0
                        print("-----------succeeded in storing model---------")
                    #else:
                    #    if epoch >= 20:
                    #        stop_step += 1
                    #        if stop_step > self.early_stop_step:
                    #            print("-------------accuracy of development-----------", dev_accuracy, self.best_dev_accuracy)
                    #            stop_flag = True
                    #            break
                    print("-------------accuracy of development-----------", dev_accuracy, self.best_dev_accuracy, dev_loss)
            print("--------accuracy of training----------", train_loss/float(train_internal_cnt), train_accuracy/float(train_internal_cnt), epoch)
                            
            # if stop_step > self.early_stop_step and stop_flag == True:
            #     break
                        
                
                
            
            
    

