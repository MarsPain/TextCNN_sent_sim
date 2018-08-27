from collections import Counter
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_dict(data):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param data:从原CSV中读取的训练数据，[index,"能不能开花呗老兄","花呗逾期了还能开通",label（0/1）]
    :return:四个dict:word和label与索引index之间的双向映射字典
    """
    word_to_index = {}
    index_to_word = {}
    label_to_index = {'0':0, '1':1}
    index_to_label = {0:'0', 1:'1'}
    word_to_index["_PAD"] = 0
    index_to_word[0] = "_PAD"
    word_to_index["_UNK"] = 1
    index_to_word[1] = "_UNK"
    c_inputs = Counter()    # Counter用于统计字符串里某个字符出现的次数
    vocab_list = []  # 存储高词频的word及其相应的频数
    for i, row in enumerate(data):
        string_list_1 = list(row[1])
        string_list_2 = list(row[2])
        c_inputs.update(string_list_1)
        c_inputs.update(string_list_2)
        vocab_list = c_inputs.most_common(20000)  # 参数对word数量进行限制
    for i, word_freq in enumerate(vocab_list):
        # print(word_freq)  # word_freq是word和相应词频的元组
        word, _ = word_freq
        word_to_index[word] = i + 2
        index_to_word[i+2] = word
    return word_to_index, index_to_word, label_to_index, index_to_label


def create_data(data):
    pass


def get_features(data):  # 特征工程，创建和获取各个类型的特征：tfidf值、词向量、编辑距离等
    # 获取tfidf值
    tfidf_path = "data/tfidf.txt"   # 存储tfidf值的文件路径
    if not os.path.exists(tfidf_path):
        get_tfidf_and_save(data, tfidf_path)
    tfidf_dict = load_tfidf_dict(tfidf_path)
    # 获取fasttext词向量
    fasttext_path = "data/fasttext_fin_model_50.vec"
    fasttext_dict = load_vector(fasttext_path)
    # 获取word2vec词向量
    word2vec_path = "data/word2vec.txt"
    word2vec_dict = load_vector(word2vec_path)


def get_tfidf_and_save(data, tfidf_path):
    corpus = []  # 用于生成tfidf值的语料库
    for i, row in enumerate(data):
        str1 = " ".join(jieba.cut(row[1]))
        str2 = " ".join(jieba.cut(row[2]))
        corpus.append(str1)
        corpus.append(str2)
    # print("string_list:", string_list)
    tfidfvectorizer = TfidfVectorizer()
    tfidf = tfidfvectorizer.fit_transform(corpus)
    # print(tfidf)
    tfidf = tfidfvectorizer.idf_
    # print(tfidf)
    word_to_tfidf = dict(zip(tfidfvectorizer.get_feature_names(), tfidf))
    # print(word_to_tfidf)
    with open(tfidf_path, "w", encoding="utf-8") as f:
        for word, tfidf_score in word_to_tfidf.items():
            # print(k)
            f.write(word+"|||"+str(tfidf_score)+"\n")


def load_tfidf_dict(tfidf_path):
    tfidf_dict = {}
    with open(tfidf_path, "r", encoding="utf-8") as f:
        for line in f:
            word, tfidf_score = line.strip().split("|||")
            tfidf_dict[word] = float(tfidf_score)
    # print("tfidf_dict:", tfidf_dict)
    return tfidf_dict


def load_vector(path):
    vector_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and 'word2vec' in path:
                continue
            word_vec = line.strip().split()
            vec_list = [float(x) for x in word_vec[1:]]
            vector_dict[word_vec[0]] = np.asarray(vec_list)
    print("vector_dict:", vector_dict)
    return vector_dict
