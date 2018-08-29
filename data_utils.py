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
    return word_to_index, index_to_word, label_to_index, index_to_label


def features_engineer(data, word_to_index, fasttext_dict, word2vec_dict, tfidf_dict, n_gram):
    """
    特征工程，基于tfidf值、fasttext词向量、word2vec词向量构造多种特征
    :param data:
    :param word_to_index:word-index的映射字典
    :param fasttext_dict:word-fasttext字典
    :param word2vec_dict:word-word2vec字典
    :param tfidf_dict:word-tfidf字典
    :param n_gram:n_gram similiarity窗口大小
    :return:
    """
    features_vector = []
    for index, row in enumerate(data):
        features_vector_line = []
        # 此处不用，但是在拿到特征工程结果后会用到
        # word_list_1 = list(row[1])  # 第一个句子的字符word组成的列表
        # index_list_1 = [word_to_index.get(word, UNK_ID) for word in word_list_1]
        # word_list_2 = list(row[2])  # 第一个句子的字符word组成的列表
        # index_list_2 = [word_to_index.get(word, UNK_ID) for word in word_list_2]
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
        dist_word2vec_list = distance_vector_tfidf(string_list_1, string_list_2, fasttext_dict, tfidf_dict)
        features_vector_line.extend(dist_fasttext_list)
        features_vector_line.extend(dist_word2vec_list)
        # print("features_vector_line:", len(features_vector_line), features_vector_line)
        features_vector.append(features_vector_line)
    print("features_vector:", len(features_vector), features_vector)
    return features_vector


def split_string_as_list_by_ngram(input_string, ngram_value):
    """
    根据不同ngram_value大小对string进行拆分
    :param input_string: 需要拆分的字符串string
    :param ngram_value: n_gram窗口大小
    :return:
    """
    input_string = "".join([string for string in input_string if string.strip()])
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
    # input_list1 = [input_string_x1[token] for token in range(len(input_string_x1)) if input_string_x1[token].strip()]
    # input_list2 = [input_string_x2[token] for token in range(len(input_string_x2)) if input_string_x2[token].strip()]
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
