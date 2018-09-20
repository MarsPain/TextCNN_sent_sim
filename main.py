import tensorflow as tf
import numpy as np
from model import TextCNN
import os
import csv
import json
from collections import OrderedDict
import pickle
from data_utils import create_dict, features_engineer, sentence_word_to_index, shuffle_padding_split, BatchManager, init_weights_dict,\
    get_weights_for_current_batch, compute_confuse_matrix, write_predict_error_to_file, compute_labels_weights,\
    get_weights_label_as_standard_dict
from utils import get_tfidf_and_save, load_tfidf_dict, load_vector, load_word_embedding

FLAGS = tf.app.flags.FLAGS
# 文件路径参数
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt", "checkpoint location for the model")
tf.app.flags.DEFINE_string("pkl_dir", "pkl", "dir for save pkl file")
tf.app.flags.DEFINE_string("config_file", "config", "dir for save pkl file")
tf.app.flags.DEFINE_string("traning_data_path", "data/atec_nlp_sim_train.csv", "path of traning data.")
# tf.app.flags.DEFINE_string("traning_data_path", "data/atec_nlp_sim_train_demo.csv", "path of demo data.")
tf.app.flags.DEFINE_string("word2vec_model_path", "data/word2vec.txt", "word2vec's embedding for word")
# tf.app.flags.DEFINE_string("word2vec_model_path", "data/wiki_100.utf8", "word2vec's embedding for char")
tf.app.flags.DEFINE_string("fasttext_model_path", "data/fasttext_fin_model_50.vec", "fasttext's vocabulary and vectors")
# 模型参数
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 50, "number of epochs to run.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_string("tokenize_style", 'word', "tokenize sentence in char,word,or pinyin.default is char")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_integer("num_filters", 64, "number of filters")  # 64
tf.app.flags.DEFINE_integer("sentence_len", 39, "max sentence length. length should be divide by 3,""which is used by k max pooling.")
tf.app.flags.DEFINE_integer("top_k", 1, "value of top k for k-max polling")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")  # 0.001
tf.app.flags.DEFINE_boolean("decay_lr_flag", True, "whether manally decay lr")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("clip_gradients", 3.0, "clip_gradients")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability")
filter_sizes = [2, 3, 4]


class Main:
    def __init__(self):
        self.word_to_index = None   # word到index的映射字典
        self.index_to_word = None   # index到字符word的映射字典
        self.label_to_index = None   # label到index的映射字典
        self.index_to_label = None  # index到label的映射字典
        self.vocab_size = None  # 字符的词典大小
        self.num_classes = None  # 类别标签数量
        self.features_vector_size = None    # 特征工程得到的特征向量的维度
        self.train_batch_manager = None  # train数据batch生成类
        self.valid_batch_manager = None  # valid数据batch生成类
        self.test_batch_manager = None  # test数据batch生成类

    def config_model(self):
        config = OrderedDict()
        config["learning_rate"] = FLAGS.learning_rate
        config["num_classes"] = self.num_classes
        config["sequence_length"] = FLAGS.sentence_len
        config["vocab_size"] = self.vocab_size
        config["embed_size"] = FLAGS.embed_size
        config["is_training"] = FLAGS.is_training
        config["filter_sizes"] = filter_sizes
        config["num_filters"] = FLAGS.num_filters
        config["top_k"] = FLAGS.top_k
        config["features_vector_size"] = self.features_vector_size
        config["decay_steps"] = FLAGS.decay_steps
        config["decay_rate"] = FLAGS.decay_rate
        config["clip_gradients"] = FLAGS.clip_gradients
        return config

    def get_dict(self):
        """
        获取word和label与索引index之间的双向映射字典
        :return:
        """
        if not os.path.isdir(FLAGS.pkl_dir):   # 创建存储临时字典数据的目录
            os.makedirs(FLAGS.pkl_dir)
        word_label_dict = os.path.join(FLAGS.pkl_dir, "word_label_dict.pkl")    # 存储word和label与index之间的双向映射字典
        if os.path.exists(word_label_dict):  # 若word_label_path已存在
            with open(word_label_dict, 'rb') as dict_f:
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = pickle.load(dict_f)
        else:   # 重新读取训练数据并创建各个映射字典
            with open(FLAGS.traning_data_path, "r", encoding="utf-8") as data_f:
                train_data = csv.reader(data_f, delimiter='\t', quotechar='|')
                self.word_to_index, self.index_to_word, self.label_to_index, self.index_to_label = \
                    create_dict(train_data, word_label_dict, FLAGS.tokenize_style)
                # print(self.word_to_index)
        self.vocab_size = len(self.word_to_index)
        self.num_classes = len(self.label_to_index)

    def get_data(self):
        train_valid_test = os.path.join(FLAGS.pkl_dir, "train_valid_test.pkl")
        if os.path.exists(train_valid_test):    # 若train_valid_test已被处理和存储
            with open(train_valid_test, 'rb') as data_f:
                train_data, valid_data, test_data, true_label_pert = pickle.load(data_f)
        else:   # 读取数据集并创建训练集、验证集和测试集
            with open(FLAGS.traning_data_path, "r", encoding="utf-8") as data_f:
                all_data = csv.reader(data_f, delimiter='\t', quotechar='|')
                # 获取tfidf值
                tfidf_path = "data/tfidf.txt"   # 存储tfidf值的文件路径
                if not os.path.exists(tfidf_path):
                    get_tfidf_and_save(all_data, tfidf_path)
                tfidf_dict = load_tfidf_dict(tfidf_path)
                # 获取fasttext词向量
                fasttext_path = "data/fasttext_fin_model_50.vec"
                fasttext_dict = load_vector(fasttext_path)
                # 获取word2vec词向量
                word2vec_path = "data/word2vec.txt"
                word2vec_dict = load_vector(word2vec_path)
            with open(FLAGS.traning_data_path, "r", encoding="utf-8") as data_f:
                all_data = csv.reader(data_f, delimiter='\t', quotechar='|')
                # 基于句子的长度和包含的词汇、tfidf值、fasttext词向量、word2vec词向量进行特征工程，并获取相应的特征向量
                features_vector = features_engineer(all_data, fasttext_dict, word2vec_dict, tfidf_dict, FLAGS.tokenize_style, n_gram=8)
            with open(FLAGS.traning_data_path, "r", encoding="utf-8") as data_f:
                all_data = csv.reader(data_f, delimiter='\t', quotechar='|')
                # 语句序列化，将句子中的word映射成index，作为输入特征
                sentences_1, sentences_2, labels = sentence_word_to_index(all_data, self.word_to_index, self.label_to_index, FLAGS.tokenize_style)
                """
                打乱数据、padding、添加features_vector到数据中并根据比例分割成train、valid、test数据，
                train、valid、test里面又依次包含sentences_1，sentences_2，features_vector，labels四种数据
                """
                train_data, valid_data, test_data, true_label_pert = shuffle_padding_split(sentences_1, sentences_2, labels, features_vector, train_valid_test, FLAGS.sentence_len)
        self.features_vector_size = len(train_data[2][0])
        # print("features_vector_size:", self.features_vector_size)
        print("训练集大小：", len(train_data[0]), "验证集大小：", len(valid_data[0]), "正样本比例：", true_label_pert)
        # 获取train、valid、test数据的batch生成类
        self.train_batch_manager = BatchManager(train_data, int(FLAGS.batch_size))
        print("训练集批次数量：", self.train_batch_manager.len_data)
        self.valid_batch_manager = BatchManager(valid_data, int(FLAGS.batch_size))
        self.test_batch_manager = BatchManager(test_data, int(FLAGS.batch_size))

    def train(self):
        config = self.config_model()
        with open(FLAGS.config_file, "w", encoding="utf8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            text_cnn, saver = self.create_model(sess, config)
            curr_epoch = sess.run(text_cnn.epoch_step)
            iteration = 0
            best_acc = 0.60
            best_f1_score = 0.20
            weights_dict = init_weights_dict(self.label_to_index)   # 初始化类别权重参数矩阵
            for epoch in range(curr_epoch, FLAGS.num_epochs):
                loss, eval_acc, counter = 0.0, 0.0, 0
                # train
                for batch in self.train_batch_manager.iter_batch(shuffle=True):
                    iteration += 1
                    input_x1, input_x2, features_vector, input_y = batch
                    weights = get_weights_for_current_batch(input_y, weights_dict)   # 更新类别权重参数矩阵
                    feed_dict = {text_cnn.input_x1: input_x1, text_cnn.input_x2: input_x2, text_cnn.features_vector: features_vector, text_cnn.input_y: input_y,
                                 text_cnn.weights: weights, text_cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                 text_cnn.iter: iteration, text_cnn.tst: not FLAGS.is_training}
                    curr_loss, curr_acc, lr, _ = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.learning_rate, text_cnn.train_op], feed_dict)
                    loss, eval_acc, counter = loss+curr_loss, eval_acc+curr_acc, counter+1
                    if counter % 100 == 0:  # steps_check
                        print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tAcc:%.3f\tLearning rate:%.5f" % (epoch, counter, loss/float(counter), eval_acc/float(counter), lr))
                print("going to increment epoch counter....")
                sess.run(text_cnn.epoch_increment)
                # valid
                if epoch % FLAGS.validate_every == 0:
                    eval_loss, eval_accc, f1_scoree, precision, recall, weights_label = self.evaluate(sess, text_cnn, self.valid_batch_manager, iteration)
                    weights_dict = get_weights_label_as_standard_dict(weights_label)
                    print("label accuracy(used for label weight):==========>>>>", weights_dict)
                    print("【Validation】Epoch %d\t Loss:%.3f\tAcc %.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f" % (epoch, eval_loss, eval_accc, f1_scoree, precision, recall))
                    # save model to checkpoint
                    if f1_scoree > best_f1_score:
                        save_path = FLAGS.ckpt_dir + "/model.ckpt"
                        print("going to save model. eval_f1_score:", f1_scoree, ";previous best f1 score:", best_f1_score,
                              ";eval_acc", str(eval_accc), ";previous best_acc:", str(best_acc))
                        saver.save(sess, save_path, global_step=epoch)
                        best_acc = eval_accc
                        best_f1_score = f1_scoree
                    # if FLAGS.decay_lr_flag and (epoch != 0 and (epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40)):
                    #     for i in range(2):  # decay learning rate if necessary.
                    #         print(i, "Going to decay learning rate by half.")
                    #         sess.run(text_cnn.learning_rate_decay_half_op)
            # test
            test_loss, acc_t, f1_score_t, precision, recall, weights_label = self.evaluate(sess, text_cnn, self.valid_batch_manager, iteration)
            print("Test Loss:%.3f\tAcc:%.3f\tF1 Score:%.3f\tPrecision:%.3f\tRecall:%.3f:" % (test_loss, acc_t, f1_score_t, precision, recall))

    def create_model(self, sess, config):
        text_cnn = TextCNN(config)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            if FLAGS.decay_lr_flag:
                for i in range(2):  # decay learning rate if necessary.
                    print(i, "Going to decay learning rate by half.")
                    sess.run(text_cnn.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(FLAGS.ckpt_dir):
                os.makedirs(FLAGS.ckpt_dir)
            if FLAGS.use_pretrained_embedding:  # 加载预训练的词向量
                print("===>>>going to use pretrained word embeddings...")
                old_emb_matrix = sess.run(text_cnn.Embedding.read_value())
                new_emb_matrix = load_word_embedding(old_emb_matrix, FLAGS.word2vec_model_path, FLAGS.embed_size, self.index_to_word)
                word_embedding = tf.constant(new_emb_matrix, dtype=tf.float32)  # 转为tensor
                t_assign_embedding = tf.assign(text_cnn.Embedding, word_embedding)  # 将word_embedding复制给text_cnn.Embedding
                sess.run(t_assign_embedding)
                print("using pre-trained word emebedding.ended...")
        return text_cnn, saver

    def evaluate(self, sess, text_cnn, batch_manager, iteration):
        small_value = 0.00001
        file_object = open('data/log_predict_error.txt', 'a')
        eval_loss, eval_accc, eval_counter = 0.0, 0.0, 0
        eval_true_positive, eval_false_positive, eval_true_negative, eval_false_negative = 0, 0, 0, 0
        weights_label = {}  # weight_label[label_index]=(number,correct)
        for batch in batch_manager.iter_batch(shuffle=True):
            eval_x1, eval_x2, features_vector, eval_y = batch
            weights = np.ones(len(eval_x1))   # weights的shape要与batch对上
            feed_dict = {text_cnn.input_x1: eval_x1, text_cnn.input_x2: eval_x2, text_cnn.features_vector: features_vector, text_cnn.input_y: eval_y,
                         text_cnn.weights: weights, text_cnn.dropout_keep_prob: 1.0, text_cnn.iter: iteration, text_cnn.tst: True}
            curr_eval_loss, curr_accc, logits = sess.run([text_cnn.loss_val, text_cnn.accuracy, text_cnn.logits], feed_dict)
            true_positive, false_positive, true_negative, false_negative = compute_confuse_matrix(logits, eval_y)
            write_predict_error_to_file(file_object, logits, eval_y, self.index_to_word, eval_x1, eval_x2)
            eval_loss, eval_accc, eval_counter = eval_loss+curr_eval_loss, eval_accc+curr_accc, eval_counter+1
            eval_true_positive, eval_false_positive = eval_true_positive+true_positive, eval_false_positive+false_positive
            eval_true_negative, eval_false_negative = eval_true_negative+true_negative, eval_false_negative+false_negative
            weights_label = compute_labels_weights(weights_label, logits, eval_y)
        print("true_positive:", eval_true_positive, ";false_positive:", eval_false_positive, ";true_negative:", eval_true_negative, ";false_negative:", eval_false_negative)
        p = float(eval_true_positive)/float(eval_true_positive+eval_false_positive+small_value)
        r = float(eval_true_positive)/float(eval_true_positive+eval_false_negative+small_value)
        f1_score = (2*p*r)/(p+r+small_value)
        print("eval_counter:", eval_counter, ";eval_acc:", eval_accc)
        return eval_loss/float(eval_counter), eval_accc/float(eval_counter), f1_score, p, r, weights_label

if __name__ == "__main__":
    main = Main()
    main.get_dict()
    main.get_data()
    main.train()
