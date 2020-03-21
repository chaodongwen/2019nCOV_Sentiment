import fasttext
import sys
import csv


# 获取txt文件的每一行并写入list中
def get_lines_from_txt(txt_dir):
    with open(txt_dir, "r", encoding="utf-8") as f:
        text_list = [text.strip('\n') for text in f.readlines()]
    return text_list


# 模型训练
def model_train(processed_train_data, model_save_dir):
    print("model_training" + "--" * 50)
    # 统计训练集中正面(PS)中立(MS)负面(NS)样本的个数
    PS, MS, NS = 0, 0, 0
    with open(processed_train_data, "r", encoding="utf-8") as f:
        for text in f.readlines():
            if (text.split("__")[-1]).split(" ")[0] == "1":
                PS += 1
            elif (text.split("__")[-1]).split(" ")[0] == "-1":
                NS += 1
            else:
                MS += 1
    print("训练集中正面样本个数为:", PS)
    print("训练集中立样本个数为:", MS)
    print("训练集中负面样本个数为:", NS)
    print("训练集样本总个数为:", (PS + MS + NS))

    # 训练监督文本，thread表示以几个线程进行训练，不加默认1个线程
    classifier = fasttext.train_supervised(input=processed_train_data, label_prefix='__label__', thread=4)
    # 保存模型
    classifier.save_model(model_save_dir)


# 模型测试
def model_test(test_data_save_dir, model_load_dir):
    print("model_testing" + "--"*50)
    # 载入模型
    classifier = fasttext.load_model(model_load_dir)
    test_data_list = get_lines_from_txt(test_data_save_dir)
    result_dict = {}
    for i in range(len(test_data_list)):
        wbid = test_data_list[i].split("__")[0]
        text = test_data_list[i].split("__")[1]
        # 进行预测
        predict = classifier.predict(text)
        y = str(predict[0][0].split('__')[-1])
        # 将预测结果与微博id一起装进字典里
        result_dict[wbid] = y
    return result_dict


if __name__ == "__main__":
    processed_train_data = "./data_preprocess/processed_data/processed_uncut_train.txt"
    processed_test_data = "./data_preprocess/processed_data/processed_uncut_test.txt"
    model_save_dir = "fasttext_model/text_classifier_base_uncut.model"
    csv_file_save_dir = "data_preprocess/processed_data/backup/test_result_fasttext_uncut.csv"
    # 训练
    model_train(processed_train_data, model_save_dir)

    # 预测
    model_load_dir = "fasttext_model/text_classifier_base_uncut.model"
    result_dict = model_test(processed_test_data, model_load_dir)
    # 写入csv文件
    with open(csv_file_save_dir, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "y"])
        for key, value in result_dict.items():
            writer.writerow([key, value])
    print("finish!")

