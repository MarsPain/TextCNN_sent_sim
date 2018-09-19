from collections import Counter
import os
import jieba
import codecs
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_and_save(data, tfidf_path):
    """
    获取tfidf值并写入到文件中
    :param data:
    :param tfidf_path:
    :return:
    """
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
    """
    加载tfidf值
    :param tfidf_path:word-tfidf的映射字典
    :return:
    """
    tfidf_dict = {}
    with open(tfidf_path, "r", encoding="utf-8") as f:
        for line in f:
            word, tfidf_score = line.strip().split("|||")
            tfidf_dict[word] = float(tfidf_score)
    # print("tfidf_dict:", tfidf_dict)
    return tfidf_dict


def load_vector(path):
    """
    加载词向量
    :param path:预训练的词向量文件路径
    :return:word-词向量的映射字典
    """
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


def load_word_embedding(emb_matrix, word2vec_model_path, embed_size, index_to_word):
        print("Loading pretrained embeddings from", word2vec_model_path)
        pre_trained = {}
        emb_invalid = 0
        for i, line in enumerate(codecs.open(word2vec_model_path, 'r', 'utf-8')):
            line = line.rstrip().split()
            if len(line) == embed_size + 1:
                pre_trained[line[0]] = np.asarray([float(x) for x in line[1:]]).astype(np.float32)
            else:
                emb_invalid += 1
        if emb_invalid > 0:
            print('WARNING: %i invalid lines' % emb_invalid)
        # print(pre_trained)
        c_found = 0
        c_lower = 0
        c_zeros = 0
        n_words = len(index_to_word)
        for i in range(n_words):
            word = index_to_word[i]
            # print(word)
            if word in pre_trained:
                # print("YES!!!!!!")
                emb_matrix[i] = pre_trained[word]
                c_found += 1
            elif word.lower() in pre_trained:
                emb_matrix[i] = pre_trained[word.lower()]
                c_lower += 1
            elif re.sub('\d', '0', word.lower()) in pre_trained:
                emb_matrix[i] = pre_trained[
                    re.sub('\d', '0', word.lower())
                ]
                c_zeros += 1
        print('Loaded %i pretrained embeddings.' % len(pre_trained))
        print('%i / %i (%.4f%%) words have been initialized with ''pretrained embeddings.' %
              (c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))
        print('%i found directly, %i after lowercasing, ''%i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
        return emb_matrix
