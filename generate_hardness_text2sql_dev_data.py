import os
import torch
import argparse
from tqdm import tqdm
import json

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generate hardness text2sql dev data.")
    parser.add_argument('--save_path', type=str, default='./data/preprocessed_data/preprocessed_by_resd_hardness/Easy_spider_dev_dataset.json',
                        help='output json file save path.')
    parser.add_argument('--input_resd_hardness_file', type=str, default='./data/preprocessed_data/preprocessed_by_resd_hardness/Easy_resd_hardness_dev_dataset.json',
                        help='input batch size.')
    parser.add_argument('--input_spider_file', type=str, default='./data/spider/dev.json',
                        help='input batch size.')
    opt = parser.parse_args()
    return opt

def generate(opt):
    with open(opt.input_resd_hardness_file, 'r') as resd_f:
        resd_dataset = json.load(resd_f)

    with open(opt.input_spider_file, 'r') as spider_f:
        spider_dataset = json.load(spider_f)

    # print(resd_dataset[192])

    hardness_spider_dataset = []
    count = 0
    count_a = 0
    for resd_data in tqdm(resd_dataset):
        count += 1
        db_id_resd = resd_data["db_id"]
        input_sequence_resd = resd_data["input_sequence"].split(" |")
        question_resd = input_sequence_resd[0]
        for spider_data in spider_dataset:
            db_id_spider = spider_data["db_id"]
            question_spider = spider_data["question"].replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", "'").replace(
                "\u201d", "'").strip()

            if db_id_spider == db_id_resd and question_spider == question_resd:
                count_a += 1
                hardness_spider_dataset.append(spider_data)
                # if count is not count_a:
                #     print(count)
                    # print("question_spider:", question_spider)
                    # print("question_resd:", question_resd)
            else:
                continue
    print("RESD_Dev_sample_number:", len(resd_dataset))
    print("Hardness_Spider_Dev_sample_number:", len(hardness_spider_dataset))
    if len(resd_dataset) == len(hardness_spider_dataset):
        print("Success!!!: The number of hardness_dev sample is qual to resd_hrdness")
        with open(opt.save_path, "w") as f:
            json_str = json.dumps(hardness_spider_dataset, indent=2)
            f.write(json_str)
        print("Save in :", opt.save_path)
    else:
        print("Error: The number of hardness_dev sample is NOT qual to resd_hrdness")

if __name__ == "__main__":
    opt = parse_option()
    generate(opt)
    print("Finished!")
