import os
import json
import torch
import argparse
import torch.optim as optim
import transformers
import pandas as pd

from tqdm import tqdm
from tokenizers import AddedToken
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generate hardness text2sql dev data.")
    parser.add_argument('--easy_model_sql', type=str,
                        default='Easy_DQCP_3B_predicted_natsql.txt', help='output json file save path.')
    # parser.add_argument('--medium_model_sql', type=str,
    #                     default='Medium_DQCP_3B_predicted_natsql.txt', help='output json file save path.')
    # parser.add_argument('--hard_model_sql', type=str,
    #                     default='Hard_DQCP_3B_predicted_natsql.txt', help='output json file save path.')
    # parser.add_argument('--extra_hard_model_sql', type=str,
    #                     default='Extra-hard_DQCP_3B_predicted_natsql.txt', help='output json file save path.')
    parser.add_argument('--easy_original_dev_filepath', type=str,
                        default='./data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Easy_spider_dev_dataset.json',
                        help='output json file save path.')
    # parser.add_argument('--medium_original_dev_filepath', type=str,
    #                     default='./data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Medium_spider_dev_dataset.json',
    #                     help='output json file save path.')
    # parser.add_argument('--hard_original_dev_filepath', type=str,
    #                     default='./data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Hard_spider_dev_dataset.json',
    #                     help='output json file save path.')
    # parser.add_argument('--extra_original_dev_filepath', type=str,
    #                     default='./data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Extra-hard_spider_dev_dataset.json',
    #                     help='output json file save path.')
    parser.add_argument('--gold_path', type=str,
                        default='./DQCP_gold_dev_natsql.txt',
                        help='output json file save path.')
    parser.add_argument('--pred_path', type=str,
                        default='./DQCP_predict_dev_natsql.txt',
                        help='output json file save path.')
    parser.add_argument('--db_path', type=str, default='./database')

    opt = parser.parse_args()
    return opt


def evaluator(opt):
    # ------------load output json file----------------
    with open(opt.easy_model_sql, 'r', encoding='utf-8') as f1:
        line = f1.readline()
        easy_sql = []
        while line:
            easy_sql.append(line)
            line = f1.readline()
    # with open(opt.medium_model_sql, 'r', encoding='utf-8') as f2:
    #     line = f2.readline()
    #     medium_sql = []
    #     while line:
    #         medium_sql.append(line)
    #         line = f2.readline()
    # with open(opt.hard_model_sql, 'r', encoding='utf-8') as f3:
    #     line = f3.readline()
    #     hard_sql = []
    #     while line:
    #         hard_sql.append(line)
    #         line = f3.readline()
    # with open(opt.extra_hard_model_sql, 'r', encoding='utf-8') as f4:
    #     line = f4.readline()
    #     extra_sql = []
    #     while line:
    #         extra_sql.append(line)
    #         line = f4.readline()

    predict_sqls = []
    predict_sqls.extend(easy_sql)
    # predict_sqls.extend(medium_sql)
    # predict_sqls.extend(hard_sql)
    # predict_sqls.extend(extra_sql)

    # output the predict sql
    predict_path = opt.pred_path
    with open(predict_path, "w") as f:
        for line in predict_sqls:
            f.write(line + '\n')
    f.close()

    # -------------- build original dev file --------------------------
    with open(opt.easy_original_dev_filepath, 'r', encoding='utf-8') as f_e:
        easy_original_dev = json.load(f_e)
    # with open(opt.medium_original_dev_filepath, 'r', encoding='utf-8') as f_m:
    #     medium_original_dev = json.load(f_m)
    # with open(opt.hard_original_dev_filepath, 'r', encoding='utf-8') as f_h:
    #     hard_original_dev = json.load(f_h)
    # with open(opt.extra_original_dev_filepath, 'r', encoding='utf-8') as f_ex:
    #     extra_original_dev = json.load(f_ex)

    original_dev = []
    original_dev.extend(easy_original_dev)
    # original_dev.extend(medium_original_dev)
    # original_dev.extend(hard_original_dev)
    # original_dev.extend(extra_original_dev)

    Dev_path = './DQCP_orginal_spider_dev.json'
    with open(Dev_path, "w") as f:
        original_dev_str = json.dumps(original_dev, indent = 2)
        f.write(original_dev_str)

    # ------------- evaluate --------------------------
    # initialize evaluator
    evaluator = EvaluateTool()
    evaluator.register_golds(Dev_path, opt.db_path)
    # output the gold sql
    golds = []
    for gold_sample in evaluator.golds:
        gold_query = gold_sample['query']
        gold_db_id = gold_sample['db_id']
        print(gold_query + '\t' +gold_db_id)
        golds.append(gold_query + '\t' +gold_db_id)

    gold_path = opt.gold_path
    # gold_path = './DQCP_gold_dev_natsql.txt'
    with open(gold_path, "w") as f:
        for line in golds:
            f.write(line + '\n'+ '\n')
    f.close()

    spider_metric_result = evaluator.evaluate(predict_sqls)
    print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
    print('exec score: {}'.format(spider_metric_result["exec"]))
    print("---------------------------------------------")
    print('easy score: {}'.format(spider_metric_result["easy"] / 248))
    print('medium score: {}'.format(spider_metric_result["medium"] / 446))
    print('hard score: {}'.format(spider_metric_result["hard"] / 174))
    print('extra-hard score: {}'.format(spider_metric_result["extra"] / 166))
    return spider_metric_result["exact_match"], spider_metric_result["exec"]

    # -------------------- display -------------------


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    evaluator(opt)