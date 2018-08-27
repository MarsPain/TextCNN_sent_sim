from collections import Counter


def create_dict(data, path):
    """
    通过训练集创建字符word和label与索引index之间的双向映射字典
    :param data:被读取的训练数据
    :param path:存储字典的路径
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
    for i, row in enumerate(data):
        string_list_1 = list(row[1])
        string_list_2 = list(row[2])
        c_inputs.update(string_list_1)
        c_inputs.update(string_list_2)
        vocab_list = c_inputs.most_common(20000)  # 找到词频最高的字符并排序，参数对映射字典大小进行限制
    for i, word_freq in enumerate(vocab_list):
        # print(word_freq)  # word_freq是word和相应词频的元组
        word, _ = word_freq
        word_to_index[word] = i + 2
        index_to_word[i+2] = word
    return word_to_index, index_to_word, label_to_index, index_to_label
