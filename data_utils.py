from collections import Counter


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
