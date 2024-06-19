from calcuate_metric import calculate_oral_metric, calculate_cherrant_metric, calculate_baseline_metric
from pandas import DataFrame
import numpy as np
import os
import argparse
import json
import pickle
import Levenshtein
from pypinyin import pinyin,Style
from transformers import AutoTokenizer
import jieba
from collections import defaultdict

def get_path(folder_path):
    """
    @param folder_path : the root path of json files
    @return names: the name of file under the root path
    """
    file_names = os.listdir(folder_path)
    sorted_file_names = sorted(file_names)
    names = sorted_file_names

    return names

def write2excel(results,header_row,report_path):
    """
    @param results: 2D lists
    @param header_row: the head of excel
    @param report_path: the place to save the excel
    """
    shuchus = np.array(results)
    shuchu_dict = {}

    for i in range(len(header_row)):
        shuchu_dict[header_row[i]] = list(shuchus[:,i])

    df = DataFrame(shuchu_dict)
    df.to_excel(report_path, index=False)

def calculate_oral_metic2excel(args, mode = "difflib"):
    """
    @param path: the path of files that will be tested
    @param report_path: the path to save the excel
    """
    result_path = args.base_dir + "all_result/"
    names = get_path(result_path)

    results = []
    for name in names:
        name_all = result_path + name
        if mode == "difflib": result = [name] + calculate_oral_metric(name_all, args.base_dir)
        elif mode == "cherrant": result = [name] + calculate_cherrant_metric(name_all, args.base_dir)
        elif mode == "baseline": result = [name] + calculate_baseline_metric(name_all, args.base_dir)
        results.append(result)

    write2excel(results,args.header_row, args.base_dir+f'mid_result/analyse_metric_{mode}.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str,
                        default="")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--header_row", type=list, default=['name','S_D_p','S_D_r', 'S_D_f1', 'S_C_p', 'S_C_r', 'S_C_f1', 'C_D_p','C_D_r', 'C_D_f1', 'C_C_p', 'C_C_r', 'C_C_f1'])
    parser.add_argument("--base_dir", type=str, default="./predicts/prediction_qwen_14b/") 
    parser.add_argument("--mode", type=str, default="cherrant") #cherrant / difflib
    args = parser.parse_args()
    
    calculate_oral_metic2excel(args, mode = args.mode)

    