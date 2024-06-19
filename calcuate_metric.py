import json
import pickle
import os
import difflib

def calculate_metric(src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars=""):
    """
    :param src_sentences: list of origin sentences
    :param tgt_sentences: list of target sentences
    :param pred_sentences: list of predict sentences
    :param report_file: report file path
    :param ignore_chars: chars that is not evaluated
    :return:
    """
    src_char_list, tgt_char_list, pred_char_list = input_check_and_process(src_sentences, tgt_sentences, pred_sentences)
    sentence_detection, sentence_correction, char_detection, char_correction = [
        {'all_error': 0, 'true_predict': 0, 'all_predict': 0} for _ in range(4)]
    output_errors = []
    for src_chars, tgt_chars, pred_chars in zip(src_char_list, tgt_char_list, pred_char_list):
        true_error_indexes = []
        detect_indexes = []
        for index, (src_char, tgt_char, pred_char) in enumerate(zip(src_chars, tgt_chars, pred_chars)):
            if src_char in ignore_chars:
                src_chars[index] = tgt_char
                pred_chars[index] = tgt_char
                continue
            if src_char != tgt_char:
                char_detection['all_error'] += 1
                char_correction['all_error'] += 1
                true_error_indexes.append(index)
            if src_char != pred_char:
                char_detection['all_predict'] += 1
                char_correction['all_predict'] += 1
                detect_indexes.append(index)
                if src_char != tgt_char:
                    char_detection['true_predict'] += 1
                if pred_char == tgt_char:
                    char_correction['true_predict'] += 1
        if true_error_indexes:
            sentence_detection['all_error'] += 1
            sentence_correction['all_error'] += 1
        if detect_indexes:
            sentence_detection['all_predict'] += 1
            sentence_correction['all_predict'] += 1
            if tuple(true_error_indexes) == tuple(detect_indexes):
                sentence_detection['true_predict'] += 1
            if tuple(tgt_chars) == tuple(pred_chars):
                sentence_correction['true_predict'] += 1
        if tuple(tgt_chars) != tuple(pred_chars):
            origin_s = "".join(src_chars)
            target_s = "".join(tgt_chars)
            predict_s = "".join(pred_chars)
            if origin_s == target_s and origin_s != predict_s:
                error_type = "过纠"
            elif origin_s != target_s and origin_s == predict_s:
                error_type = "漏纠"
            else:
                error_type = '综合'
            output_errors.append(
                [
                    "原始: " + "".join(src_chars),
                    "正确: " + "".join([c2 if c1 == c2 else f"【{c2}】" for c1, c2 in zip(pred_chars, tgt_chars)]),
                    "预测: " + "".join([c1 if c1 == c2 else f"【{c1}】" for c1, c2 in zip(pred_chars, tgt_chars)]),
                    "错误类型: " + error_type
                ]
            )

    # result = dict()
    # for prefix_name, sub_metric in zip(['S_D_', 'S_C_', 'C_D_', 'C_C_'],
    #                                    [sentence_detection, sentence_correction, char_detection, char_correction]):
    #     sub_r = compute_p_r_f1(sub_metric['true_predict'], sub_metric['all_predict'], sub_metric['all_error']).items()
    #     for k, v in sub_r:
    #         result[prefix_name + k] = v
    # if report_file:
    #     write_report(report_file, result, output_errors)
    # return result

    result = dict()
    result_list = []
    # for prefix_name, sub_metric in zip(['C_D_', 'C_C_'],
    #                                    [char_detection, char_correction]):
    for prefix_name, sub_metric in zip(['S_D_', 'S_C_', 'C_D_', 'C_C_'],
                                   [sentence_detection, sentence_correction, char_detection, char_correction]):
        sub_r = compute_p_r_f1(sub_metric['true_predict'], sub_metric['all_predict'], sub_metric['all_error']).items()
        for k, v in sub_r:
            result[prefix_name + k] = v
            result_list.append(v)
    if report_file:
        write_report(report_file, result, output_errors)
    return result,result_list

def input_process(data):
    src_sentences, tgt_sentences, pred_sentences = [], [], []

    if 'predict' in data[0]:
        for i in range(len(data)):
            a = chuli(data[i]['src'],data[i]['predict'])
            s,t = data[i]['src'],data[i]['tgt']
            if len(s) == len(t) == len(a):pass
            else: 
                print(i,len(s),len(t),len(a))
                print(s)
                print(t)
                print(a)
                assert False
            src_sentences.append(s)
            tgt_sentences.append(t)
            pred_sentences.append(a)
    return src_sentences, tgt_sentences, pred_sentences

def process_txt(name, base_dir):
    with open(name, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    fold_dir = base_dir + "mid_result/" + name.split("/")[-1].split(".")[0]
    with open(fold_dir + "_pred.txt", "wt",encoding="utf-8") as f:
        for id in range(len(data)):
            key = 'predict'
            pred = data[id][key]
            f.write(f"{pred}\n")

    with open(fold_dir + "_src.txt", "wt",encoding="utf-8") as f:
        for id in range(len(data)):
            src = data[id]['src']
            f.write(f"{src}\n")
             
    count_empty = 0 
    with open(fold_dir + "_src_pred.txt", "wt",encoding="utf-8") as f:
        for id in range(len(data)):
            key = 'predict'
            if data[id][key] != "":
                w = data[id][key].find("纠错后的句子：")
                ww = data[id][key].find("\n")
                if  w != -1 :
                    src,pred = data[id]['src'],data[id][key][w+7:]
                    f.write(f"{id+1}\t{src}\t{pred}\n")
                elif ww != -1:
                    if ww != 0: src,pred = data[id]['src'],data[id][key][:ww]
                    else: src,pred = data[id]['src'],data[id][key][ww+1:]
                    f.write(f"{id+1}\t{src}\t{pred}\n")
                else:
                    src,pred = data[id]['src'],data[id][key]
                    f.write(f"{id+1}\t{src}\t{pred}\n")
            else: 
                count_empty += 1
                assert False, data[id]
                src = data[id]['src']
                f.write(f"{id+1}\t{src}\t{src}\n")
        print("count_empty: ",count_empty)
        
    os.chdir("./MuCGEC/scorers/ChERRANT")
    HYP_PARA_FILE = fold_dir + "_src_pred.txt"
    HYP_M2_FILE= fold_dir + "_src_pred_op"
    os.system(f'python parallel_to_m2.py -f {HYP_PARA_FILE} -o {HYP_M2_FILE} -g word')

def write_op(name, base_dir):

    fold_dir = base_dir + "mid_result/" + name.split("/")[-1].split(".")[0]
    with open(fold_dir + "_src_pred_op", 'r', encoding='utf-8') as f:  # 过滤op文件
        lines = f.readlines()
        new_lines = []
        flag = True
        for i,line in enumerate(lines):
            line = line.strip("\n")
            if line and str(line).find("T0-A1") != -1: flag = False
            if flag:
                new_lines.append(line)
            if line == "" and flag == False: 
                new_lines.append("")
                flag = True

    with open(fold_dir + "_src_pred_op", 'w', encoding='utf-8') as f:  # 重写op文件
        for line in new_lines:
            f.write(line + '\n')

    with open(fold_dir + "_src_pred_op", 'r', encoding='utf-8') as f:
        id = 0
        dics,op_s = [],[]
        dic = {}
        lines = f.readlines()
        print(len(lines))
        for i,line in enumerate(lines):
            dic['id'] = id
            line = line.strip("\n")
            if line: 
                # print(line[0])
                # print("line",line)
                if line[0] == "S": dic['src'] = "".join(line[2:].split(" "))
                if line[0] == "T": 
                    if str(line).find("没有错误") != -1: dic['pred'] = dic['src']
                    else: dic['pred'] = "".join(line[6:].split(" "))
                if line[0] == "A":
                    l = line[2:].split("|||")[:3]
                    op_s.append(l)
                # print(line)
            else:
                id += 1
                dic["op"] = op_s
                dics.append(dic)
                dic = {}
                op_s = []
    with open(fold_dir + "_ops.pkl","wb") as f:
        pickle.dump(dics,f)

def str_full_to_half(full_str):
    half_str = ""
    for char in full_str:
        inside_code = ord(char)
        if inside_code == 12288:  # 全角空格直接转化
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        half_str += chr(inside_code)
    return half_str

def is_chinese_char(c):
    """
    判断字符是否是汉字
    :param c:
    :return:
    """
    if len(c) > 1:
        return False
    return '\u4e00' <= c <= '\u9fa5'

def is_chinese_string(s):
    """
    检查是否字符串全是汉字
    :param s: 输入字符串
    :return:
    """
    return all(is_chinese_char(c) for c in s)

def chuli(origin, corrected, is_print=False):
    """
    make origin and corrected sentence lengths consistent
    """
    origin_list = list(origin) 
    s = difflib.SequenceMatcher(None, origin, corrected)
    
    for tag, i1,i2,j1, j2 in s.get_opcodes():
        if tag == 'replace' and (i2 - i1) == (j2 - j1) and is_chinese_string(origin_list[i1:i2]):
            origin_list[i1:i2] = list(corrected[j1:j2])
    corrected = "".join(origin_list)          
    if is_print: 
        print('=' * 10 + str(not origin == corrected) + '=' * 10) 
        print(origin) 
        print(corrected) 
        print('='*20) 
    
    return corrected

def cherrant_chuli(name_b, base_dir):

    print("="*5,name_b,"="*5)
    fold_dir = base_dir + "mid_result/" + name_b.split("/")[-1].split(".")[0]
    # assert False
    
    if os.path.exists(fold_dir + "_ops.pkl"): pass
    else:
        process_txt(name_b, base_dir)
        write_op(name_b, base_dir)
    
    with open(fold_dir + "_ops.pkl", mode='rb') as f:
        data_preds = pickle.load(f)

    #读tgt
    with open(name_b, "r", encoding="utf-8") as fh:
        data_tgts = json.load(fh)

    print(len(data_preds),len(data_tgts))
    assert len(data_preds)==len(data_tgts)
    
    src_sentences, tgt_sentences, pred_sentences = [], [], []

    for index,data_pred in enumerate(data_preds):

        src_chars = list(data_tgts[index]['src'])
        tgt_chars = list(data_tgts[index]['tgt'])
        # pred_chars = list(data_pred['pred'])

        for ops in data_pred['op']:

            #pos_start是改的原句的位置起点, pos_end是改的原句的位置终点, op是操作字符（"w""S""noop""R""M"）, val(tgt改后的字符)
            pos_start, pos_end, op, val = eval(ops[0].split(" ")[0]), eval(ops[0].split(" ")[1]), ops[1], ops[2].split(" ")
            
            # op操作计数
            src_word = str_full_to_half("".join(src_chars[pos_start: pos_end]))
            if op == "S" and pos_end - pos_start == len(val): # and ("".join(val)).find("UNK") == -1 and (src_word).find("“") == -1 and (src_word).find("”") == -1 and src_word.isascii() != True and (src_word).find("�") == -1 and (src_word).find("…") == -1: 
                src_chars[pos_start:pos_end] = val
        
        pred_sen = "".join(src_chars)
                
        assert len(pred_sen) == len(src_chars) == len(tgt_chars),(len(pred_sen),len(src_chars),len(tgt_chars),index,src_chars,pred_sen)

        src_sentences.append(data_tgts[index]['src'])
        tgt_sentences.append(data_tgts[index]['tgt'])
        pred_sentences.append(pred_sen)
    return src_sentences, tgt_sentences, pred_sentences

def write_report(output_file, metric, output_errors):
    """
    generate report
    @param output_file:
    @param metric:
    @param output_errors:
    @return:
    """
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('overview:\n')
        for key in metric:
            f.write(f'{key}:{metric[key]}\n')
        f.write('\nbad cases:\n')
        for output_error in output_errors:
            f.write("\n".join(output_error))
            f.write("\n\n")

def compute_p_r_f1(true_predict, all_predict, all_error):
    """
    @param true_predict:
    @param all_predict:
    @param all_error:
    @return:
    """
    p = round(true_predict / all_predict * 100, 3)
    r = round(true_predict / all_error * 100, 3)
    f1 = round(2 * p * r / (p + r + 1e-10), 3)
    return {'p': p, 'r': r, 'f1': f1}

def input_check_and_process(src_sentences, tgt_sentences, pred_sentences):
    """
    check the input is valid
    @param src_sentences:
    @param tgt_sentences:
    @param pred_sentences:
    @return:
    """
    assert len(src_sentences) == len(tgt_sentences) == len(pred_sentences)
    src_char_list = [list(s) for s in src_sentences]
    tgt_char_list = [list(s) for s in tgt_sentences]
    pred_char_list = [list(s) for s in pred_sentences]
    assert all(
        [len(src) == len(tgt) == len(pred) for src, tgt, pred in zip(src_char_list, tgt_char_list, pred_char_list)]
    )
    return src_char_list, tgt_char_list, pred_char_list

def calculate_oral_metric(name,path):

    # print("baocuo : ",name)
    with open(name, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    src_sentences, tgt_sentences, pred_sentences = input_process(data)
    print("="*5,name,"="*5)
    name = name.split("/")[-1].split(".")[0]
    prf,prf_list = calculate_metric(src_sentences, tgt_sentences, pred_sentences, path + "mid_result/"+ f'{name}_prf.txt')
    print(prf)

    return prf_list

def calculate_cherrant_metric(name, base_dir):

    src_sentences, tgt_sentences, pred_sentences = cherrant_chuli(name, base_dir)
    print(len(src_sentences))
    assert len(src_sentences) == len(tgt_sentences) == len(pred_sentences)

    print("="*5,name,"="*5)
    name = name.split("/")[-1].split(".")[0]
    prf,prf_list = calculate_metric(src_sentences, tgt_sentences, pred_sentences, base_dir + "mid_result/" + f'{name}_prf_cherrant.txt')
    print(prf)

    return prf_list
