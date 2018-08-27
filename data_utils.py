from collections import Counter
import numpy as np

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"


def create_dict(data):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param data:从原CSV中读取的训练数据，[index,"能不能开花呗老兄","花呗逾期了还能开通",label（0/1）]
    :return:四个dict:word和label与索引index之间的双向映射字典
    """
    word_to_index = {}
    index_to_word = {}
    label_to_index = {'0': 0, '1': 1}
    index_to_label = {0: '0', 1: '1'}
    word_to_index[_PAD] = PAD_ID
    index_to_word[PAD_ID] = _PAD
    word_to_index[_UNK] = UNK_ID
    index_to_word[UNK_ID] = _UNK
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


def features_engineer(data, word_to_index, fasttext_dict, word2vec_dict, tfidf_dict, n_gram):
    features_vector = []
    for _, row in enumerate(data):
        features_vector_line = []
        # 此处不用，但是在拿到特征工程结果后悔用到
        # word_list_1 = list(row[1])  # 第一个句子的字符word组成的列表
        # index_list_1 = [word_to_index.get(word, UNK_ID) for word in word_list_1]
        # word_list_2 = list(row[2])  # 第一个句子的字符word组成的列表
        # index_list_2 = [word_to_index.get(word, UNK_ID) for word in word_list_2]
        string_1 = row[1]
        string_2 = row[2]
        # 获取n-gram similiarity
        for i in range(n_gram):
            x1_list = split_string_as_list_by_ngram(string_1, i+1)
            x2_list = split_string_as_list_by_ngram(string_2, i + 1)
            ngram_sim_1 = compute_ngram_sim(x1_list, x2_list)
            ngram_sim_2 = compute_ngram_sim(x2_list, x1_list)
            features_vector_line.append(ngram_sim_1)
            features_vector_line.append(ngram_sim_2)


def split_string_as_list_by_ngram(input_string, ngram_value):
    input_string = "".join([string for string in input_string if string.strip()])
    length = len(input_string)
    result_string = []
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i+ngram_value])
    # print("ngram:",ngram_value,"result_string:",result_string)
    return result_string


def compute_ngram_sim(x1_list, x2_list):    # 计算n-gram similiarity（blue score）
    count_dict = {}
    count_dict_clip = {}
    # 1. count for each token at predict sentence side.
    for token in x1_list:
        count_dict[token] = count_dict[token] + 1 if token in count_dict else 1
        # if token not in count_dict:
        #     count_dict[token] = 1
        # else:
        #     count_dict[token] = count_dict[token] + 1
    count = np.sum([value for key, value in count_dict.items()])
    # 2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            count_dict_clip[token] = count_dict_clip[token] + 1 if token in count_dict_clip else 1
            # if token not in count_dict_clip:
            #     count_dict_clip[token] = 1
            # else:
            #     count_dict_clip[token] = count_dict_clip[token] + 1
    # 3. clip value to ceiling value for that token
    count_dict_clip = {key: (value if value <= count_dict[key] else count_dict[key])
                       for key, value in count_dict_clip.items()}
    count_clip = np.sum([value for key, value in count_dict_clip.items()])
    result = float(count_clip)/(float(count)+0.00000001)
    return result
