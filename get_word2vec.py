import csv
import re
import jieba
from gensim.models import word2vec

path_data_all = "data/atec_nlp_sim_train.csv"
path_word2vec_word_string = "data/word2vec_word_string.txt"   # 用于训练word的word2vec的语料库
path_word2vec_char_string = "data/word2vec_char_string.txt"   # 用于训练char的word2vec的语料库


def get_word_data():
    with open(path_data_all, "r", encoding="utf-8") as data_f:
        all_data = csv.reader(data_f, delimiter='\t', quotechar='|')
        string = ""
        for i, row in enumerate(all_data):
            string_1 = row[1].strip()
            string_1 = re.sub("\*+", "*", string_1)
            string_2 = row[2].strip()
            string_2 = re.sub("\*+", "*", string_2)
            string += string_1 + "。" + string_2 + "。"
    jieba.suggest_freq('花呗', True)
    jieba.suggest_freq('借呗', True)
    jieba.suggest_freq('还款日', True)
    jieba.suggest_freq('代还', True)
    string_list = jieba.cut(string)
    string = " ".join(string_list)
    with open(path_word2vec_word_string, "w", encoding="utf-8") as f:
        f.write(string)


def get_char_data():
    with open(path_data_all, "r", encoding="utf-8") as data_f:
        all_data = csv.reader(data_f, delimiter='\t', quotechar='|')
        string_list = []
        for i, row in enumerate(all_data):
            string_list_1 = list(row[1].strip()) + ["。"]
            string_list_2 = list(row[2].strip()) + ["。"]
            string_list += string_list_1 + string_list_2
    string = " ".join(string_list)
    with open(path_word2vec_char_string, "w", encoding="utf-8") as f:
        f.write(string)


def get_word2vec(type):
    if type == "word":
        sentences = word2vec.LineSentence(path_word2vec_word_string)
        model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=4, size=100, iter=20)  # CBOW
        model.save("data/word2vec_word_model")
        model.wv.save_word2vec_format('data/word2vec_word_model.txt', binary=False)
    else:
        sentences = word2vec.LineSentence(path_word2vec_char_string)
        model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=4, size=100, iter=20)  # CBOW
        model.save("data/word2vec_char_model")
        model.wv.save_word2vec_format('data/word2vec_char_model.txt', binary=False)


if __name__ == "__main__":
    get_word_data()   # 词嵌入
    get_word2vec("word")
    get_char_data()   # 字嵌入
    get_word2vec("char")
