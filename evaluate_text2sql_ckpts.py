import argparse
import os
import json

from text2sql import _test

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for selecting the best ckpt.")
    
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'input batch size.')
    parser.add_argument('--device', type = str, default = "0",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/DQCP-text2sql-t5-large-Easy",
                        help = 'save path of fine-tuned text2sql models.')
    parser.add_argument('--eval_results_path', type = str, default = "./Easy_model_eval_results/resd_hardness_text2sql-t5-large-Easy",
                        help = 'the evaluation results of fine-tuned text2sql models.')
    parser.add_argument('--mode', type = str, default = "eval",
                        help='eval.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/preprocessed_data/preprocessed_by_resd_hardness/Easy_resd_hardness_dev_dataset.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "./data/preprocessed_data/preprocessed_by_resd_hardness/Easy_spider_dev_dataset.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "./database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    
    opt = parser.parse_args()

    return opt

    
if __name__ == "__main__":
    opt = parse_option()
    
    ckpt_names = os.listdir(opt.save_path)
    ckpt_names = sorted(ckpt_names, key = lambda x:eval(x.split("-")[1]))
    
    print("model_ckpt_names:", ckpt_names)

    save_path = opt.save_path
    os.makedirs(opt.eval_results_path, exist_ok = True)

    eval_results = []
    for ckpt_name in ckpt_names:
        # if the current ckpt is being evaluated or has already been evaluated
        if "{}.txt".format(ckpt_name) in os.listdir(opt.eval_results_path):
            # is being evaluated
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "r") as f:
                if len(f.readlines()) == 1:
                    continue
            # has already been evaluated
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "r") as f:
                eval_result = json.load(f)
                eval_results.append(eval_result)
        # otherwise, we start evaluating the current ckpt
        else:
            print("Start evaluating ckpt: {}".format(ckpt_name))
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "w") as f:
                f.write("Evaluating...")
            
            opt.save_path = save_path + "/{}".format(ckpt_name)
            em, exec = _test(opt)   # Kernel !!!
            
            eval_result = dict()
            eval_result["ckpt"] = opt.save_path
            eval_result["EM"] = em
            eval_result["EXEC"] = exec

            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "w") as f:
                f.write(json.dumps(eval_result, indent=2))
            
            eval_results.append(eval_result)
    
    for eval_result in eval_results:
        print("ckpt name:", eval_result["ckpt"])
        print("EM:", eval_result["EM"])
        print("EXEC:", eval_result["EXEC"])
        print("-----------")

    em_list = [er["EM"] for er in eval_results]
    exec_list = [er["EXEC"] for er in eval_results]
    em_and_exec_list = [em + exec for em, exec in zip(em_list, exec_list)]

    # find best EM ckpt
    best_em, exec_in_best_em = 0.00, 0.00
    best_em_idx = 0

    # find best EXEC ckpt
    best_exec, em_in_best_exec = 0.00, 0.00
    best_exec_idx = 0

    # find best EM + EXEC ckpt
    best_em_plus_exec = 0.00
    best_em_plus_exec_idx = 0

    for idx, (em, exec) in enumerate(zip(em_list, exec_list)):
        if em > best_em or (em == best_em and exec > exec_in_best_em):
            best_em = em
            exec_in_best_em = exec
            best_em_idx = idx
        
        if exec > best_exec or (exec == best_exec and em > em_in_best_exec):
            best_exec = exec
            em_in_best_exec = em
            best_exec_idx = idx
        
        if em+exec > best_em_plus_exec:
            best_em_plus_exec = em+exec
            best_em_plus_exec_idx = idx
    
    print("Best EM ckpt:", eval_results[best_em_idx])
    print("Best EXEC ckpt:", eval_results[best_exec_idx])
    print("Best EM+EXEC ckpt:", eval_results[best_em_plus_exec_idx])