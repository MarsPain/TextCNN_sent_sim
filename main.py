import tensorflow as tf
import numpy as np
# from model import TextCNN
# from data_util import create_vocabulary,load_data
import os
import csv
import random
import pickle
# from weight_boosting import compute_labels_weights,get_weights_for_current_batch,get_weights_label_as_standard_dict,init_weights_dict
import gensim
from gensim.models import KeyedVectors
from utils import create_dict, dump_dict

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("tokenize_style", 'word', "the style of tokenize sentence in char or word. default is char")
tf.app.flags.DEFINE_string("dict_dir", "model", "name scope value.")
tf.app.flags.DEFINE_boolean("decay_lr_flag", True, "whether manally decay lr")
tf.app.flags.DEFINE_integer("embed_size", 50, "embedding size")  # 128
tf.app.flags.DEFINE_integer("num_filters", 64, "number of filters")  # 64
tf.app.flags.DEFINE_integer("sentence_len", 39, "max sentence length. length should be divide by 3, "
                                                "which is used by k max pooling.")
tf.app.flags.DEFINE_string("similiarity_strategy", 'additive', "similiarity strategy: additive or multiply. "
                                                               "default is additive")  # to tackle miss typed words
tf.app.flags.DEFINE_string("max_pooling_style", 'chunk_max_pooling',
                           "max_pooling_style:max_pooling,k_max_pooling,chunk_max_pooling. default: chunk_max_pooling")
tf.app.flags.DEFINE_integer("top_k", 3, "value of top k")
tf.app.flags.DEFINE_string("traning_data_path", "data/atec_nlp_sim_train.csv", "path of traning data.")
tf.app.flags.DEFINE_integer("vocab_size", 13422, "maximum vocab size.")  # 80000
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")  # 0.001
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 20, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding", False, "whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path", "data/fasttext_fin_model_50.vec", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability")
filter_sizes = [2, 3, 4]


class Main:
    def __init__(self):
        self.word_to_index = None
        self.index_to_word = None
        self.label_to_index = None
        self.index_to_label = None

    def get_dict(self):
        if not os.path.isdir(FLAGS.dict_dir):   # 创建存储临时字典数据的目录
            os.makedirs(FLAGS.dict_dir)
        word_label_path = os.path.join(FLAGS.dict_dir, "word_label.pkl")    # 存储word和label与index之间的双向映射字典
        if os.path.exists(word_label_path):  # 若word_label_path已存在
            with open(word_label_path, 'rb') as dict_f:
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = pickle.load(dict_f)
        else:   # 重新读取训练数据并创建各个映射字典
            with open(FLAGS.traning_data_path, "r", encoding="utf-8") as data_f:
                train_data = csv.reader(data_f, delimiter='\t', quotechar='|')
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = create_dict(train_data)
        # print(self.word_to_index)

    def get_data(self):
        pass

    def get_batch_data(self):
        pass

if __name__ == "__main__":
    main = Main()
    main.get_dict()
