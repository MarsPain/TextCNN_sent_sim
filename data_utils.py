from collections import Counter
import numpy as np
import pickle
import math
import random

PAD_ID = 0
UNK_ID = 1
_PAD = "_PAD"
_UNK = "UNK"
valid_num = 1600
test_num = 800


def create_dict(data, path):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param data:从原CSV中读取的训练数据，[index,"能不能开花呗老兄","花呗逾期了还能开通",label（0/1）]
    :param path:存储生成的映射字典的路径
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
        string_list_1 = list(row[1].strip())
        string_list_2 = list(row[2].strip())
        c_inputs.update(string_list_1)
        c_inputs.update(string_list_2)
        vocab_list = c_inputs.most_common(20000)  # 参数对word数量进行限制
    for i, word_freq in enumerate(vocab_list):
        # print(word_freq)  # word_freq是word和相应词频的元组
        word, _ = word_freq
        word_to_index[word] = i + 2
        index_to_word[i+2] = word
    with open(path, "wb") as dict_f:  # 创建映射字典后进行存储
        pickle.dump([word_to_index, index_to_word, label_to_index, index_to_label], dict_f)
    return word_to_index, index_to_word, label_to_index, index_to_label


def features_engineer(data, fasttext_dict, word2vec_dict, tfidf_dict, n_gram):
    """
    特征工程，基于tfidf值、fasttext词向量、word2vec词向量构造多种特征
    :param data:
    :param fasttext_dict:word-fasttext字典
    :param word2vec_dict:word-word2vec字典
    :param tfidf_dict:word-tfidf字典
    :param n_gram:n_gram similiarity窗口大小
    :return:
    """
    features_vector = []
    for index, row in enumerate(data):
        features_vector_line = []
        string_1 = row[1].strip()
        string_2 = row[2].strip()
        # 获取n-gram similarity
        for i in range(n_gram):
            x1_list = split_string_as_list_by_ngram(string_1, i+1)
            x2_list = split_string_as_list_by_ngram(string_2, i + 1)
            ngram_sim_1 = compute_ngram_sim(x1_list, x2_list)
            ngram_sim_2 = compute_ngram_sim(x2_list, x1_list)
            # print("ngram_sim_1:", ngram_sim_1)
            features_vector_line.append(ngram_sim_1)
            features_vector_line.append(ngram_sim_2)
        # 获取两个句子的长度差异性（length difference）
        len_string_1 = float(len(string_1))
        len_string_2 = float(len(string_2))
        len_diff = (float(abs(len_string_1-len_string_2)))/((len_string_1+len_string_2)/2.0)
        features_vector_line.append(len_diff)
        # 获取单词相似度以及差异度（相同词和不同词的比例）
        sentence_diff_list = get_sentence_diff(index, string_1, string_2)
        features_vector_line.extend(sentence_diff_list)
        # 获取编辑距离
        edit_dist = float(get_edit_distance(string_1, string_2))/30.0
        features_vector_line.append(edit_dist)
        # 基于词向量以及tfidf计算文本的余弦距离、曼哈登距离等
        string_list_1 = list(string_1)
        string_list_2 = list(string_2)
        dist_fasttext_list = distance_vector_tfidf(string_list_1, string_list_2, fasttext_dict, tfidf_dict)
        dist_word2vec_list = distance_vector_tfidf(string_list_1, string_list_2, word2vec_dict, tfidf_dict)
        features_vector_line.extend(dist_fasttext_list)
        features_vector_line.extend(dist_word2vec_list)
        # print("features_vector_line:", len(features_vector_line), features_vector_line)
        features_vector.append(features_vector_line)
    # print("features_vector:", len(features_vector), features_vector)
    return features_vector


def split_string_as_list_by_ngram(string, ngram_value):
    """
    根据不同ngram_value大小对string进行拆分
    :param string: 需要拆分的字符串string
    :param ngram_value: n_gram窗口大小
    :return:
    """
    input_string = "".join([string for string in string if string.strip()])
    length = len(input_string)
    result_string = []
    for i in range(length):
        if i + ngram_value < length + 1:
            result_string.append(input_string[i:i+ngram_value])
    # print("ngram:",ngram_value,"result_string:",result_string)
    return result_string


def compute_ngram_sim(x1_list, x2_list):
    """
    计算n-gram similiarity
    :param x1_list:
    :param x2_list:
    :return:
    """
    count_dict = {}
    count_dict_clip = {}
    # 1. count for each token at predict sentence side.
    for token in x1_list:
        count_dict[token] = count_dict[token] + 1 if token in count_dict else 1
    count = np.sum([value for key, value in count_dict.items()])
    # 2.count for tokens existing in predict sentence for target sentence side.
    for token in x2_list:
        if token in count_dict:
            count_dict_clip[token] = count_dict_clip[token] + 1 if token in count_dict_clip else 1
    # 3. clip value to ceiling value for that token
    count_dict_clip = {key: (value if value <= count_dict[key] else count_dict[key])
                       for key, value in count_dict_clip.items()}
    count_clip = np.sum([value for key, value in count_dict_clip.items()])
    result = float(count_clip)/(float(count)+0.00000001)
    return result


def get_sentence_diff(index, string_1, string_2):
    """
    获取单词相似度以及差异度（相同词和不同词的比例）
    :param index:
    :param string_1:第一个句子
    :param string_2:第二个句子
    :return:
    """
    string_list_1 = list(string_1)
    string_list_2 = list(string_2)
    length1 = len(string_list_1)
    length2 = len(string_list_2)
    num_same = 0
    same_word_list = []
    # 计算相同的词在句子中所占比例
    for word1 in string_list_1:
        for word2 in string_list_2:
            if word1 == word2:
                num_same += 1
                same_word_list.append(word1)
                continue
    num_same_pert_min = float(num_same)/float(max(length1, length2))
    num_same_pert_max = float(num_same) / float(min(length1, length2))
    num_same_pert_avg = float(num_same) / (float(length1+length2)/2.0)
    # 计算不同的词在句子中所占比例
    input_list1_unique = set([x for x in string_list_1 if x not in same_word_list])
    input_list2_unique = set([x for x in string_list_2 if x not in same_word_list])
    num_diff_x1 = float(len(input_list1_unique))/float(length1)
    num_diff_x2 = float(len(input_list2_unique)) / float(length2)
    if index == 0:  # print debug message
        pass
        # print("string_1:", string_1)
        # print("string_2:", string_2)
    sentence_diff_list = [num_same_pert_min, num_same_pert_max, num_same_pert_avg, num_diff_x1, num_diff_x2]
    return sentence_diff_list


def get_edit_distance(string_1, string_2):
    """
    获取两个句子之间的编辑距离
    :param string_1:
    :param string_2:
    :return:
    """
    matrix = [[i + j for j in range(len(string_2) + 1)] for i in range(len(string_1) + 1)]  # 动态规划矩阵
    for i in range(1, len(string_1) + 1):
        for j in range(1, len(string_2) + 1):
            if string_1[i - 1] == string_2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    return matrix[len(string_1)][len(string_2)]


def distance_vector_tfidf(string_list_1, string_list_2, vector_dict, tfidf_dict, tfidf_flag=True):
    """
    基于词向量以及tfidf计算文本的余弦距离、曼哈登距离等
    :param string_list_1:第一个句子中的单词组成的列表
    :param string_list_2:第二个句子中的单词组成的列表
    :param vector_dict:词向量的映射字典，键值对为word-vector
    :param tfidf_dict:tfidf值的映射字典，键值对为word-tfidf score
    :param tfidf_flag:是否在计算中利用tfidf值
    :return:
    """
    # 从词向量字典中获取词向量
    sentence_vec_1 = get_sentence_vector(vector_dict, tfidf_dict, string_list_1, tfidf_flag=tfidf_flag)
    sentence_vec_2 = get_sentence_vector(vector_dict, tfidf_dict, string_list_2, tfidf_flag=tfidf_flag)
    # 计算余弦相似度
    numerator = np.sum(np.multiply(sentence_vec_1, sentence_vec_2))
    denominator = np.sqrt(np.sum(np.power(sentence_vec_1, 2)))*np.sqrt(np.sum(np.power(sentence_vec_2, 2)))
    cos_distance = float(numerator)/float(denominator+0.000001)
    # 计算曼哈顿距离
    manhattan_distance = np.sum(np.abs(np.subtract(sentence_vec_1, sentence_vec_2)))
    if np.isnan(manhattan_distance):
        manhattan_distance = 300.0
    manhattan_distance = np.log(manhattan_distance+0.000001)/5.0
    # 计算堪培拉距离
    canberra_distance = np.sum(np.abs(sentence_vec_1-sentence_vec_2)/np.abs(sentence_vec_1+sentence_vec_2))
    if np.isnan(canberra_distance):
        canberra_distance = 300.0
    canberra_distance = np.log(canberra_distance+0.000001)/5.0
    # 计算明可夫斯基距离
    minkowski_distance = np.power(np.sum(np.power((sentence_vec_1-sentence_vec_2),3)), 0.33333333)
    if np.isnan(minkowski_distance):
        minkowski_distance = 300.0
    minkowski_distance = np.log(minkowski_distance+0.000001)/5.0
    # 计算欧几里得距离
    euclidean_distance = np.sqrt(np.sum(np.power((sentence_vec_1-sentence_vec_2),2)))
    if np.isnan(euclidean_distance):
        euclidean_distance = 300.0
    euclidean_distance = np.log(euclidean_distance+0.000001)/5.0
    return cos_distance, manhattan_distance, canberra_distance, minkowski_distance, euclidean_distance


def get_sentence_vector(vector_dict, tfidf_dict, string_list, tfidf_flag):
    """
    基于词向量字典、tfidf字典、语句的词列表得到每个句子的向量表示
    :param vector_dict:词向量的映射字典，键值对为word-vector
    :param tfidf_dict:tfidf值的映射字典，键值对为word-tfidf score
    :param string_list:句子中的单词组成的列表
    :param tfidf_flag:是否在计算中利用tfidf值
    :return:
    """
    vector_sentence = 0.0
    # vector_dim = len(vector_dict['花呗'])
    for word in string_list:
        word_vec = vector_dict.get(word, None)
        word_tfidf = tfidf_dict.get(word, None)
        if word_vec is None is None or word_tfidf is None:
            continue
        else:
            if tfidf_flag:
                vector_sentence += word_vec*word_tfidf
            else:
                vector_sentence += word_vec * 1.0
    vec_sentence = vector_sentence / (np.sqrt(np.sum(np.power(vector_sentence, 2)))+0.000001)
    return vec_sentence


def sentence_word_to_index(data, word_to_index, label_to_index):
    """
    根据word到index的映射字典将语句中的word转换成index
    :param data: 全数据集
    :param word_to_index:
    :param label_to_index:
    :return:
    """
    sentences_1 = []
    sentences_2 = []
    labels = []
    for index, row in enumerate(data):
        string_list_1 = list(row[1])  # 第一个句子的字符word组成的列表
        sentence_1 = [word_to_index.get(word, UNK_ID) for word in string_list_1]
        sentences_1.append(sentence_1)
        string_list_2 = list(row[2])  # 第一个句子的字符word组成的列表
        sentence_2 = [word_to_index.get(word, UNK_ID) for word in string_list_2]
        sentences_2.append(sentence_2)
        label = label_to_index[row[3]]
        labels.append(label)
    # print(sentences_1, sentences_2, labels)
    return sentences_1, sentences_2, labels


def shuffle_padding_split(sentences_1, sentences_2, labels, features_vector, path, sentence_len):
    """
    将数据集随机打乱，然后按照比例分割并生成训练集、验证集、测试集。
    :param sentences_1:
    :param sentences_2:
    :param labels:
    :param features_vector: 特征工程获得的特征向量
    :param path: dump的目标路径
    :param sentence_len: 预设的最大句子长度，将每个句子填充到这个长度
    :return:
    """
    # print(sentences_1, sentences_2, labels, features_vector)
    s_1 = []
    s_2 = []
    l = []
    f = []
    len_data = len(labels)
    random_perm = np.random.permutation(len_data)   # 对索引进行随机排序
    # print(len(random_perm))
    # print(len(features_vector))
    for index in random_perm:
        s_1.append(sentences_1[index])
        s_2.append(sentences_2[index])
        f.append(features_vector[index])
        l.append(labels[index])
    s_1 = pad_sequences(s_1, sentence_len, PAD_ID)  # padding
    s_2 = pad_sequences(s_2, sentence_len, PAD_ID)
    train_num = len_data - valid_num - test_num
    # 可以在此处通过数据增强生成更多的训练数据，比如调换sentence_1和sentence_2的位置就是一个新的样本了，
    train_data = (s_1[:train_num], s_2[:train_num], f[:train_num], l[:train_num])
    valid_data = (s_1[train_num:train_num+valid_num], s_2[train_num:train_num+valid_num],
                  f[train_num:train_num+valid_num], l[train_num:train_num+valid_num])
    test_data = (s_1[train_num+valid_num:len_data], s_2[train_num+valid_num:len_data],
                 f[train_num+valid_num:len_data], l[train_num+valid_num:len_data])
    true_label_numbers = 0   # 记录正样本比例，用于调整不同样本的权重参数
    for label in l:
        true_label_numbers += 1 if label == 1 else 0
    true_label_pert = float(true_label_numbers) / float(len_data)
    with open(path, "wb") as f:
        pickle.dump([train_data, valid_data, test_data, true_label_pert], f)
    return train_data, valid_data, test_data, true_label_pert


def pad_sequences(s, maxlen, value):
    s_new = []
    for string in s:
        if len(string) < maxlen:
            padding = [value] * (maxlen - len(string))
            s_new.append(string + padding)
        elif len(string) > maxlen:
            s_new.append(string[:maxlen])
        else:
            s_new.append(string)
    return s_new


class BatchManager(object):
    """
    用于生成batch数据的batch管理类
    """
    def __init__(self, data,  batch_size):
        self.batch_data = self.get_batch(data, batch_size)  # 根据batch_size生成所有batch数据并存入batch_data列表
        self.len_data = len(self.batch_data)    # batch数量

    @staticmethod
    def get_batch(data, batch_size):
        num_batch = int(math.ceil(len(data[0]) / batch_size))
        batch_data = list()
        for i in range(num_batch):
            # print(i)
            sentences_1 = data[0][i*batch_size:(i+1)*batch_size]
            sentences_2 = data[1][i*batch_size:(i+1)*batch_size]
            features_vector = data[2][i*batch_size:(i+1)*batch_size]
            labels = data[3][i*batch_size:(i+1)*batch_size]
            batch_data.append([sentences_1, sentences_2, features_vector, labels])
        return batch_data

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def init_weights_dict(weights_dict):
    weights_dict[1] = 0.2
    weights_dict[0] = 1.0
    return weights_dict


def get_weights_for_current_batch(answer_list, weights_dict):
    weights_list_batch = list(np.ones((len(answer_list))))
    answer_list = list(answer_list)
    for i, label in enumerate(answer_list):
        acc = weights_dict[label]
        weights_list_batch[i] = min(1.3, 1.0/(acc+0.000001))
    return weights_list_batch


def compute_confuse_matrix(logit, label):
    length = len(label)
    true_positive = 0  # TP:if label is true('1'), and predict is true('1')
    false_positive = 0  # FP:if label is false('0'),but predict is ture('1')
    true_negative = 0  # TN:if label is false('0'),and predict is false('0')
    false_negative = 0  # FN:if label is false('0'),but predict is true('1')
    for i in range(length):
        predict = np.argmax(logit[i])
        if predict == 1 and label[i] == 1:
            true_positive += 1
        elif predict == 1 and label[i] == 0:
            false_positive += 1
        elif predict == 0 and label[i] == 0:
            true_negative += 1
        elif predict == 0 and label[i] == 1:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative


def write_predict_error_to_file(file_object, logit, label, index_to_word, x1_index_list, x2_index_list):
    length = len(label)
    for i in range(length):
        predict = np.argmax(logit[i])
        if predict != label[i]:
            x1 = [index_to_word[x] for x in list(x1_index_list[i])]
            x2 = [index_to_word[x] for x in list(x2_index_list[i])]
            file_object.write("-------------------------------------------------------\n")
            file_object.write("label:"+str(label[i])+";predict:"+str(predict)+"\n")
            file_object.write("".join(x1)+"\n")
            file_object.write("".join(x2) + "\n")


def compute_labels_weights(weights_label, logits, labels):
    """
    计算每一轮epoch的类别权重参数并进行更新
    :param weights_label:
    :param logits: 预测值[None,num_classes]
    :param labels: 真实标签[None,]
    :return:
    """
    for i in range(len(labels)):
        label = labels[i]
        label_predict = np.argmax(logits[i])
        weight = weights_label.get(label, None)
        if weight == None:
            if label_predict == label:
                weights_label[label] = (1, 1)
            else:
                weights_label[label] = (1, 0)
        else:
            number = weight[0]
            correct = weight[1]
            number += 1
            if label_predict == label:
                correct += 1
            weights_label[label] = (number, correct)
    return weights_label


def get_weights_label_as_standard_dict(weights_label):
    weights_dict = {}
    for k, v in weights_label.items():
        count, correct = v
        weights_dict[k] = float(correct)/float(count)
    print("weight_dict(print accuracy):", weights_dict)
    return weights_dict
