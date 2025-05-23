import os
import json
import torch
import argparse
import torch.optim as optim
import transformers

from tqdm import tqdm
from tokenizers import AddedToken

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset, Text2SkeletonDataset, Text2SkeletonWithHdsDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "0",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 16,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/text2sql-t5-base/checkpoint-39312",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type = str, default = "tensorboard_log/Skeleton-base(test)",
                        help = 'save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type = str, default = "./models/text2sql-t5-base/checkpoint-39312",
                        help = 
                        '''
                        ./models/text2sql-t5-base/checkpoint-39312
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',#type=bool, default=True ,#
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='train, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "./data/preprocessed_data/sktsql_train_structure.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/preprocessed_data/sktsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--output", type = str, default = "predicted_skt_base",
                help = "save file of the predicted sqls.")
    parser.add_argument("--hardness_prompt", action='store_true',
                        help="Training Only. If input sample using hds prompt, act it.")
    
    opt = parser.parse_args()

    return opt

def decode_skeletons(db_path, generator_outputs, batch_db_ids, batch_inputs, tokenizer, batch_tc_original):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_skeletons = []

    for batch_id in range(batch_size):
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        for seq_id in range(num_return_sequences):
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens=True)
            pred_skeletons = pred_sequence.split("|")[-1].strip()
            pred_skeletons = pred_skeletons.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
        final_skeletons.append(pred_skeletons)

    return final_skeletons

def decode_skeletons_natsql(db_path, generator_outputs, batch_db_ids, batch_inputs, tokenizer, batch_tc_original, table_dict):
    batch_size = generator_outputs.shape[0]
    num_return_sequences = generator_outputs.shape[1]

    final_skeletons = []

    for batch_id in range(batch_size):
        db_id = batch_db_ids[batch_id]
        db_file_path = db_path + "/{}/{}.sqlite".format(db_id, db_id)
        for seq_id in range(num_return_sequences):
            pred_sequence = tokenizer.decode(generator_outputs[batch_id, seq_id, :], skip_special_tokens=True)
            pred_natsql = pred_sequence.split("|")[-1].strip()
            pred_natsql = pred_natsql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")

        final_skeletons.append(pred_natsql)

    return final_skeletons

def _train(opt):
    set_seed(opt.seed)
    print(opt)

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    
    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    if opt.hardness_prompt is not True:
        train_dataset = Text2SkeletonDataset(
            dir_=opt.train_filepath,
            mode="train"
        )
    else:
        train_dataset = Text2SkeletonWithHdsDataset(
            dir_=opt.train_filepath,
            mode="train"
        )

    train_dataloder = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    print("initializing text2skeleton model.")
    # initialize model
    model = T5ForConditionalGeneration.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("finished.")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
    num_checkpoint_steps = int( 1.42857 * len(train_dataset)/opt.batch_size) * 10

    if opt.use_adafactor:
        print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model.train()
    train_step = 0
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in tqdm(train_dataloder):
            train_step += 1
            
            batch_inputs = [data[0] for data in batch]
            batch_skeletons = [data[1] for data in batch]
            batch_db_ids = [data[2] for data in batch] # unused
            batch_tc_original = [data[3] for data in batch] # unused
            # -------------harness labels---------
            # batch_harness_labels = [data[4] for data in batch]

            # Print all input and its SQL squery
            # if epoch == 0:
            #     for batch_id in range(len(batch_inputs)):
            #         print(batch_inputs[batch_id])
            #         print(batch_skeletons[batch_id])
            #         print("----------------------")

            tokenized_inputs = text2sql_tokenizer(
                batch_inputs, 
                padding = "max_length",
                return_tensors = "pt",
                max_length = 512,
                truncation = True
            )
            
            with text2sql_tokenizer.as_target_tokenizer():
                tokenized_outputs = text2sql_tokenizer(
                    batch_skeletons,
                    padding = "max_length", 
                    return_tensors = 'pt',
                    max_length = 256,
                    truncation = True
                )
            
            encoder_input_ids = tokenized_inputs["input_ids"]
            encoder_input_attention_mask = tokenized_inputs["attention_mask"]

            decoder_labels = tokenized_outputs["input_ids"]
            decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs["attention_mask"]

            if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                decoder_labels = decoder_labels.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()
            
            model_outputs = model(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                labels = decoder_labels,
                decoder_attention_mask = decoder_attention_mask,
                return_dict = True
            )
            
            loss = model_outputs["loss"]
            loss.backward()

            if scheduler is not None:
                scheduler.step()

            if writer is not None:
                # record training loss (tensorboard)
                writer.add_scalar('train loss', loss.item(), train_step)
                # record learning rate (tensorboard)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # if train_step % num_checkpoint_steps == 0 and epoch >= 6:
            if  epoch >= 6 and train_step % num_checkpoint_steps == 0:
                print(f"At {train_step} training step, save a checkpoint.")
                os.makedirs(opt.save_path, exist_ok = True)
                model.save_pretrained(save_directory =opt.save_path + "/checkpoint-{}".format(train_step))
                text2sql_tokenizer.save_pretrained(save_directory =opt.save_path + "/checkpoint-{}".format(train_step))

def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
    
    if opt.target_type == "natsql":
        print("----Load tables for natsql :", opt.tables_for_natsql)
        tables = json.load(open(opt.tables_for_natsql, 'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SkeletonWithHdsDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    # initialize model
    model = T5ForConditionalGeneration.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_skeletons = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            if opt.target_type == "sql":
                predict_skeletons += decode_skeletons(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original
                )
            elif opt.target_type == "natsql":
                # print("----Decode NatSQL-----")
                predict_skeletons += decode_skeletons_natsql(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original, 
                    table_dict
                )
            else:
                raise ValueError()

    # process " <END> "
    for data_id, pred_skt in enumerate(predict_skeletons):
        for end in ("<= END>", "< END>", "END>"):
            if " {}".format(end) in pred_skt:
                pred_skt = pred_skt.replace(" {}".format(end), " <END>")
                predict_skeletons[data_id] = pred_skt.split("<END>")[0] + "<END> "

    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output + ".txt", "w", encoding = 'utf-8') as f:
        for pred in predict_skeletons:
            f.write(pred + "\n")
    # --------------------------------------------------------------------------
    # add prediction skeletons into input_sequence
    with open(opt.dev_filepath, "r") as f:
        dataset = json.load(f)

    for data_id, data in enumerate(dataset):
        # put skeletons into input_sequence
        # -------------------- 5/25 *** For Ablation <END> Experience --------------
        # skeletons_wo_hds = predict_skeletons[data_id]
        # skeletons_wo_hds.replace(" <END>", "")
        # data["input_sequence"] = "<START> " + skeletons_wo_hds + data["input_sequence"]
        # --------------------------------------------------------------
        # source demo
        data["input_sequence"] = "<START> " + predict_skeletons[data_id] + data["input_sequence"]
        data["pred_skeleton"] = predict_skeletons[data_id]
    with open(opt.output + ".json", 'w') as f:
        json_str = json.dumps(dataset, indent=2)
        f.write(json_str)
        print("Recording Predict Skeletons into Dev dataset have Finished !")
    # --------------------------------------------------------------------------
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))

    if opt.mode == "eval":
        gold_skeleton = []
        for _, data in enumerate(dataset):
            gold_skeleton.append(data["skeleton"])
        Sores = match_evalution(predict_skeletons, gold_skt=gold_skeleton)
        print("The skeleton match rate is :{}%".format(Sores * 100))
        return Sores

def match_evalution(predict_skt, gold_skt):
    sample_num = len(predict_skt)
    sores = 0
    # remove "<END> " => "<END>"
    for data_id, pred_skt in enumerate(predict_skt):
            if "<END> " in pred_skt:
                predict_skt[data_id] = pred_skt.replace("<END> ", "<END>")

    for i in range(sample_num):
        predict_skt_elm = predict_skt[i].split(' ')
        gold_skt_elm = gold_skt[i].split(' ')

        pred_lenth = len(predict_skt_elm)
        gold_lenth = len(gold_skt_elm)
        if pred_lenth != gold_lenth:
            sores += - abs(pred_lenth - gold_lenth) / pred_lenth
        for pred_elm in predict_skt_elm:
            if pred_elm in gold_skt_elm:
                sores += 1/pred_lenth
            else:   sores += - 1/pred_lenth
    print("Dev sample number: {}".format(i))
    return sores/sample_num

if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)
