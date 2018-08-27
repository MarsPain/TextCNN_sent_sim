from collections import Counter
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# def get_features(data):
#     # 获取tfidf值
#     tfidf_path = "data/tfidf.txt"   # 存储tfidf值的文件路径
#     if not os.path.exists(tfidf_path):
#         get_tfidf_and_save(data, tfidf_path)
#     tfidf_dict = load_tfidf_dict(tfidf_path)
#     # 获取fasttext词向量
#     fasttext_path = "data/fasttext_fin_model_50.vec"
#     fasttext_dict = load_vector(fasttext_path)
#     # 获取word2vec词向量
#     word2vec_path = "data/word2vec.txt"
#     word2vec_dict = load_vector(word2vec_path)


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
    # print("vector_dict:", vector_dict)
    return vector_dict
