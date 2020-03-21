import jieba
import numpy as np
import sys
import pandas as pd
import re


# 输出文件路径
train_data_save_dir = "./origin_data/nCoV_100k_train.labeled.xlsx"
test_data_save_dir = "./origin_data/nCov_10k_test.xlsx"
unlabeled_data_dir = "./origin_data/nCoV_900k_train.unlabeled.xlsx"
processed_train_data = "./processed_data/processed_train.txt"
processed_dev_data = "./processed_data/processed_dev.txt"
processed_test_data = "./processed_data/processed_test.txt"
processed_unlabeled_data = "./processed_data/processed_unlabeled.txt"

# 输出不切词版本数据
processed_uncut_train_data = "./processed_data/processed_uncut_train.txt"
processed_uncut_dev_data = "./processed_data/processed_uncut_dev.txt"
processed_uncut_test_data = "./processed_data/processed_uncut_test.txt"


# 使用jieba切分句子,将句子中的词语用空格隔开
def seg_sentence(sentence, stop_words):
    # jieba.load_userdict("my_jieba_dict.txt")
    pro_sentence = sentence.replace('\n', '')
    sentence_seged = list(jieba.cut(pro_sentence.strip()))
    outstr = []
    for word in sentence_seged:
        # 去除停用词
        if word not in stop_words:
            # 只保留汉字
            re_result = re.search(r"[\u4e00-\u9fa5]", word)
            if re_result:
                outstr.append(word)
    result_str = " ".join(outstr)
    return result_str


# 获取excel文件中的文本
def get_text_from_excel(train_data_dir, test_data_dir):
    train_data = pd.read_excel(train_data_dir)
    test_data = pd.read_excel(test_data_dir)
    # print(train_data.columns)
    # print(test_data.columns)
    # 获取文本和标签
    train_text_list = list(train_data["微博中文内容"])
    train_label_list = list(train_data["情感倾向"])
    test_id_list = list(test_data["微博id"])
    test_text_list = list(test_data["微博中文内容"])

    # 清洗训练数据
    index_list = []
    for i in range(len(train_label_list)):
        if str(train_label_list[i]) not in ['-1', '0', '1']:
            index_list.append(i)
    train_text_list = [train_text_list[i] for i in range(len(train_text_list)) if i not in index_list]
    train_label_list = [train_label_list[i] for i in range(len(train_label_list)) if i not in index_list]
    # print(len(train_label_list))

    # 打乱训练数据
    # 获取随机数
    random_state = np.random.get_state()
    np.random.shuffle(train_text_list)
    # 设置随机数
    np.random.set_state(random_state)
    np.random.shuffle(train_label_list)

    return train_text_list, train_label_list, test_id_list, test_text_list


def make_stop_words():
    stop_words = []
    with open("stop_word.txt", "r", encoding="utf-8") as f_words:
        for word in f_words.readlines():
            stop_words.append(word.strip('\n'))
    # 将空格加入停止词中
    stop_words += ['\u0020', '\u3000', ' ']
    return stop_words


# 构造切词版本的数据
class makeCutData(object):
    # 构建训练集
    def make_train_data(self, train_text_list, train_label_list):
        # 获取停止词
        stop_words = make_stop_words()
        # 构建训练集和验证集
        print("make_train_data" + "--" * 50)
        count = 0
        length = len(train_text_list)
        for i in range(len(train_text_list)):
            seg_result = seg_sentence(str(train_text_list[i]), stop_words=stop_words)
            label = train_label_list[i]
            if count < int(length*0.8):
                with open(processed_train_data, "a", encoding="utf-8") as f_out:
                    f_out.write("__label__" + str(label) + " " + seg_result)
                    f_out.write("\n")
            else:
                with open(processed_dev_data, "a", encoding="utf-8") as f_out:
                    f_out.write("__label__" + str(label) + " " + seg_result)
                    f_out.write("\n")
            # 输出进度
            count += 1
            for num in range(1, 51):
                if count == (num * length) // 50:
                    print("=" * num + ">" + "已切词{}%".format(2 * num))
                    if num == 50:
                        print("cut success!")

    # 构建测试集
    def make_test_data(self, test_id_list, test_text_list):
        # 获取停止词
        stop_words = make_stop_words()
        print("make_test_data" + "--" * 50)
        count = 0
        length = len(test_id_list)
        for i in range(len(test_id_list)):
            wbid = test_id_list[i]
            seg_result = seg_sentence(str(test_text_list[i]), stop_words=stop_words)
            with open(processed_test_data, "a", encoding="utf-8") as f_out:
                f_out.write(str(wbid) + "__" + seg_result)
                f_out.write("\n")
            # 输出进度
            count += 1
            for num in range(1, 51):
                if count == (num * length) // 50:
                    print("=" * num + ">" + "已切词{}%".format(2 * num))
                    if num == 50:
                        print("cut success!")

    # 构建无标签数据集
    def make_unlabeled_data(self):
        unlabeled_data = pd.read_excel(unlabeled_data_dir)
        unlabeled_id_list = list(unlabeled_data["微博id"])
        unlabeled_text_list = list(unlabeled_data["微博中文内容"])
        # 获取停止词
        stop_words = make_stop_words()
        print("make_unlabeled_data" + "--" * 50)
        count = 0
        length = len(unlabeled_id_list)
        for i in range(len(unlabeled_id_list)):
            wbid = unlabeled_id_list[i]
            seg_result = seg_sentence(str(unlabeled_text_list[i]), stop_words=stop_words)
            with open(processed_unlabeled_data, "a", encoding="utf-8") as f_out:
                f_out.write(str(wbid) + "__" + seg_result)
                f_out.write("\n")
            # 输出进度
            count += 1
            for num in range(1, 51):
                if count == (num * length) // 50:
                    print("=" * num + ">" + "已切词{}%".format(2 * num))
                    if num == 50:
                        print("cut success!")


# 构建不切词版本数据
class makeunCutData(object):
    # 构建训练集和验证集
    def make_train_data(self, train_text_list, train_label_list):
        # 只保留句子中的中文部分
        count = 0
        length = len(train_text_list)
        for i in range(len(train_text_list)):
            origin_text = str(train_text_list[i])
            label = train_label_list[i]
            text_result = origin_text
            # for word in origin_text:
            #     # if re.search(r"[\u4e00-\u9fa5]", word):
            #     text_result += word
            if count < int(length * 0.8):
                with open(processed_uncut_train_data, "a", encoding="utf-8") as f_out:
                    f_out.write("__label__" + str(label) + " " + text_result)
                    f_out.write("\n")
            else:
                with open(processed_uncut_dev_data, "a", encoding="utf-8") as f_out:
                    f_out.write("__label__" + str(label) + " " + text_result)
                    f_out.write("\n")
            # 输出进度
            count += 1
            for num in range(1, 51):
                if count == (num * length) // 50:
                    print("=" * num + ">" + "已切词{}%".format(2 * num))
                    if num == 50:
                        print("cut success!")

    # 构建测试集
    def make_test_data(self, test_id_list, test_text_list):
        count = 0
        length = len(test_id_list)
        for i in range(len(test_id_list)):
            origin_text = str(test_text_list[i])
            wbid = test_id_list[i]
            text_result = origin_text
            # for word in origin_text:
            #     if re.search(r"[\u4e00-\u9fa5]", word):
            #         text_result += word
            with open(processed_uncut_test_data, "a", encoding="utf-8") as f_out:
                f_out.write(str(wbid) + "__" + text_result)
                f_out.write("\n")
            # 输出进度
            count += 1
            for num in range(1, 51):
                if count == (num * length) // 50:
                    print("=" * num + ">" + "已切词{}%".format(2 * num))
                    if num == 50:
                        print("cut success!")


if __name__ == "__main__":
    # 获取数据
    train_text_list, train_label_list, test_id_list, test_text_list = \
        get_text_from_excel(train_data_save_dir, test_data_save_dir)

    # 构建切词版本数据
    make_cut_data = makeCutData()
    # 构建训练数据
    make_cut_data.make_train_data(train_text_list, train_label_list)
    # 构建测试数据
    make_cut_data.make_test_data(test_id_list, test_text_list)
    # 构建无标签数据
    # make_cut_data.make_unlabeled_data()

    # 构建不切词版本数据
    make_uncut_data = makeunCutData()
    make_uncut_data.make_train_data(train_text_list, train_label_list)
    make_uncut_data.make_test_data(test_id_list, test_text_list)
