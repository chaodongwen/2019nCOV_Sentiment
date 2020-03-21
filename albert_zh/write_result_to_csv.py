import csv
import pandas as pd
import sys
import numpy as np

predict_result_dir = "./2019ncov_albert_base_zh_add/test_results.tsv"
test_data_dir = "./chineseGLUEdatasets/2019ncov/processed_test.txt"
csv_result_dir = "./2019ncov_albert_base_zh_add/test_result_albert.csv"

predict_result = pd.read_csv(predict_result_dir, header=None)

# 第0个位置对应标签“-1”,第1个位置对应标签“0”,第2个位置对应标签“1”,
position_label_map = {0: -1, 1: 0, 2: 1}
label_list = []
for i in range(len(predict_result)):
    possible_list = list(predict_result.loc[i])[0].split("\t")
    label = position_label_map[np.argmax(possible_list)]
    label_list.append(label)

wbid_list = []
with open(test_data_dir, "r",encoding="utf-8") as f:
    for line in f.readlines():
        wbid = line.split("__")[0]
        wbid_list.append(wbid)

with open(csv_result_dir, "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["id", "y"])
    for i in range(len(wbid_list)):
        writer.writerow([wbid_list[i], label_list[i]])

