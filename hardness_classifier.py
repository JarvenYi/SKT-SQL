import os
import json
import torch
import transformers
import argparse
import torch.optim as optim

from tqdm import tqdm
from copy import deepcopy
from tokenizers import AddedToken
from utils.classifier_metric.evaluator import cls_metric, auc_metric
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, BertTokenizerFast
from utils.classifier_model import MyClassifier, JarvenClassifier, HardnessClassifier   #  ***add JarvenClassifier by Jarven
from utils.classifier_loss import ClassifierLoss, HardnessClassifierLoss
from transformers.trainer_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.load_dataset import HardnessClassifierDataset, HardnessClassifierDatasetForRESD

# ---------------------------------add pack------------------------------------
import os
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict, fields
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers.data.data_collator import DataCollatorForSeq2Seq
from seq2seq.utils.relation_data_collator import DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tokenizers import AddedToken
# from seq2seq.utils.args import ModelArguments
# from seq2seq.utils.picard_model_wrapper import PicardArguments, PicardLauncher, with_picard
# from seq2seq.utils.dataset import DataTrainingArguments, DataArguments # ***
# from seq2seq.utils.dataset_loader import load_dataset # ***
# from seq2seq.utils.spider import SpiderTrainer # ***
from seq2seq.preprocess.get_relation2id_dict import get_relation2id_dict

from seq2seq.model.model_utils import get_relation_t5_model, get_original_t5_model
from transformers import T5Config
import torch
import warnings
'''ignore "UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted 
samples. Use `zero_division` parameter to control this behavior."'''
warnings.filterwarnings("ignore")


from seq2seq.preprocess.choose_dataset import preprocess_by_dataset
#---------------------------------------------------------------------------------------

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning hardness classifier.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 2,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type = str, default = "0",
                        help = 'the id of used GPU device.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--gamma', type = float, default = 2.0,
                        help = 'gamma parameter in the focal loss. Recommended: [0.0-2.0].')
    parser.add_argument('--alpha', type = float, default = 0.75,
                        help = 'alpha parameter in the focal loss. Must between [0.0-1.0].')
    parser.add_argument('--epochs', type = int, default = 1,
                        help = 'training epochs.')
    parser.add_argument('--patience', type = int, default = 16,
                        help = 'patience step in early stopping. -1 means no early stopping.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql_hardness_classifier_Jarven/RoBerta-large/4Linear-resd",
                        help = 'save path of best fine-tuned model on validation set.')
    parser.add_argument('--tensorboard_save_path', type = str, default = './tensorboard_log/text2sql_hardness_classifier_Jarven',
                        help = 'save path of tensorboard log.')
    parser.add_argument('--train_filepath', type = str, default = "data/preprocessed_data/preprocessed_train_spider.json",
                        help = 'path of pre-processed training dataset.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'path of pre-processed development dataset.')
    parser.add_argument('--output_filepath', type = str, default = "data/preprocessed_data/dataset_with_hardness_pred_probs(resd-large_4Linear).json",
                        help = 'path of the output dataset (used in eval mode).')
    parser.add_argument('--model_name_or_path', type = str, default = "roberta-large",
                        help = '''pre-trained model name. t5-small or roberta-large''')
    parser.add_argument('--use_contents',  action='store_true', #default=True, #
                        help = 'whether to integrate db contents into input sequence')
    parser.add_argument('--add_fk_info',  action='store_true', #default=True, #
                        help = 'whether to add [FK] tokens into input sequence')
    parser.add_argument('--mode', type=str, default = "eval",
                        help='trian, eval or test.')
    parser.add_argument('--use_rasat', type=bool, default=False, # action='store_true',#
                        help = 'whether use rasat on cross-encoder classificer')
    parser.add_argument('--edge_type', type=str, default="Default",
                        help = 'choice: Default, DefaultWithoutSchemaEncoding, '
                               'DefaultWithoutSchemaLinking, MinType, Dependency_MinType')
    parser.add_argument('--resd_preprocessed_data', action='store_true',#type=bool, default=True,  #
                        help='whether use resd preprocessed data for hardness-classificer train and dev')
    opt = parser.parse_args()

    return opt

# ----------------------Hardness-------------------------------
def prepare_batch_inputs_and_labels(batch, tokenizer, use_rasat=True):

    def find_the_maxlength(data_list):
        max_length = 0
        for i in range(len(data_list)):
            length = len(data_list[i][0])
            if max_length < length:
                max_length = length
        return max_length
    def size_align_function(data_list, max_size=512):
        ''' Before List->tensor, we need keep some dimension '''
        # step1 : ndarray->list
        for i in range(len(data_list)):  # 1st dimension
            data_list[i] = data_list[i].tolist()
            for j in range(len(data_list[i])):  # 2nd dimension
                loop = data_list[i][j]  # .tolist()
                # feed to max size
                for _ in range(max_size):
                    if len(loop) < max_size:
                        loop.append(0)
                    elif len(loop) == max_size:
                        break
            for _ in range(max_size):
                if len(data_list[i]) < max_size:
                    data_list[i].append([0 for i in range(max_size)])
                elif len(data_list[i]) == max_size:
                    break
        return data_list

    batch_size = len(batch)
    batch_questions = [data[0] for data in batch]
    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]
    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]
    batch_data_hardness_labels = [data[6] for data in batch]
    if use_rasat:
        batch_relations_matrix = [data[5] for data in batch]
        batch_relations = []
    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []

    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]
        table_names_in_one_db = batch_table_names[batch_id]
        column_infos_in_one_db = batch_column_infos[batch_id]
        if use_rasat:
            relations_in_one_db = batch_relations_matrix[batch_id]
            batch_relations.append(relations_in_one_db)

        batch_column_number_in_each_table.append([len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []

        # make Input X : Q|T:C,C,C,C|T:C,C,C...
        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")
            
            for column_info in column_infos_in_one_db[table_id]:
                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")
            
            input_tokens = input_tokens[:-1]
        
        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(
        batch_input_tokens,
        return_tensors="pt",
        is_split_into_words = True,
        padding = "max_length",
        max_length = find_the_maxlength(batch_relations) if use_rasat else 512,  # 512
        truncation = True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []

    # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    # for batch_id in range(batch_size):
    #     word_ids = tokenized_inputs.word_ids(batch_index = batch_id)
    #
    #     aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
    #     aligned_table_labels, aligned_column_labels = [], []
    #
    #     # align question tokens
    #     for token_id, word_id in enumerate(word_ids):
    #         if word_id == 0:
    #             aligned_question_ids.append(token_id)
    #
    #     # align table names
    #     for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
    #         temp_list = []
    #         for token_id, word_id in enumerate(word_ids):
    #             if table_name_id == word_id:
    #                 temp_list.append(token_id)
    #         # if the tokenizer doesn't discard current table name
    #         if len(temp_list) != 0:
    #             aligned_table_name_ids.append(temp_list)
    #             aligned_table_labels.append(batch_table_labels[batch_id][t_id])
    #
    #     # align column names
    #     for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
    #         temp_list = []
    #         for token_id, word_id in enumerate(word_ids):
    #             if column_id == word_id:
    #                 temp_list.append(token_id)
    #         # if the tokenizer doesn't discard current column name
    #         if len(temp_list) != 0:
    #             aligned_column_info_ids.append(temp_list)
    #             aligned_column_labels.append(batch_column_labels[batch_id][c_id])
    #
    #     batch_aligned_question_ids.append(aligned_question_ids)
    #     batch_aligned_table_name_ids.append(aligned_table_name_ids)
    #     batch_aligned_column_info_ids.append(aligned_column_info_ids)
    #     batch_aligned_table_labels.append(aligned_table_labels)
    #     batch_aligned_column_labels.append(aligned_column_labels)

    # update column number in each table (because some tables and columns are discarded)
    # for batch_id in range(batch_size):
    #     if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
    #         batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][ : len(batch_aligned_table_labels[batch_id])]
    #
    #     if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
    #         truncated_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(batch_aligned_column_labels[batch_id])
    #         batch_column_number_in_each_table[batch_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]
    # batch_hardness_labels = [torch.LongTensor(hardness_labels) for hardness_labels in batch_data_hardness_labels]
    batch_hardness_labels = torch.LongTensor(batch_data_hardness_labels)
    if use_rasat:
        batch_relations = size_align_function(batch_relations, find_the_maxlength(batch_relations))
        batch_relations = torch.tensor(batch_relations)

    # batch_relations_tensor = torch.zeros(8,512,512)

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))
    # batch_relations = torch.tensor(batch_relations)
    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]
        # batch_hardness_labels = [hardness_labels.cuda() for hardness_labels in batch_hardness_labels]
        batch_hardness_labels = batch_hardness_labels.cuda()
        if use_rasat:
            batch_relations = batch_relations.cuda()

    if use_rasat:
        return encoder_input_ids, encoder_input_attention_mask, \
            batch_aligned_column_labels, batch_aligned_table_labels, \
            batch_aligned_question_ids, batch_aligned_column_info_ids, \
            batch_aligned_table_name_ids, batch_column_number_in_each_table, \
            batch_hardness_labels,\
            batch_relations
    else:
        return encoder_input_ids, encoder_input_attention_mask, \
               batch_aligned_column_labels, batch_aligned_table_labels, \
               batch_aligned_question_ids, batch_aligned_column_info_ids, \
               batch_aligned_table_name_ids, batch_column_number_in_each_table, \
               batch_hardness_labels
#--------------------RESD -Hardness Classifier -By Jarven -------------
def prepare_RESDdataset_input_and_labels(batch, tokenizer):
    def find_the_maxlength(data_list):
        max_length = 0
        for i in range(len(data_list)):
            length = len(data_list[i][0])
            if max_length < length:
                max_length = length
        return max_length

    def size_align_function(data_list, max_size=512):
        ''' Before List->tensor, we need keep some dimension '''
        # step1 : ndarray->list
        for i in range(len(data_list)):  # 1st dimension
            data_list[i] = data_list[i].tolist()
            for j in range(len(data_list[i])):  # 2nd dimension
                loop = data_list[i][j]  # .tolist()
                # feed to max size
                for _ in range(max_size):
                    if len(loop) < max_size:
                        loop.append(0)
                    elif len(loop) == max_size:
                        break
            for _ in range(max_size):
                if len(data_list[i]) < max_size:
                    data_list[i].append([0 for i in range(max_size)])
                elif len(data_list[i]) == max_size:
                    break
        return data_list

    batch_size = len(batch)
    batch_questions = [data[0] for data in batch]
    batch_data_hardness_labels = [data[1] for data in batch]
    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []

    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]
        batch_input_tokens.append(input_tokens)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(
        batch_input_tokens,
        return_tensors="pt",
        is_split_into_words=True,
        padding="max_length",
        max_length= 512,  # 512
        truncation=True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]
    # batch_hardness_labels = [torch.LongTensor(hardness_labels) for hardness_labels in batch_data_hardness_labels]
    batch_hardness_labels = torch.LongTensor(batch_data_hardness_labels)

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]
        # batch_hardness_labels = [hardness_labels.cuda() for hardness_labels in batch_hardness_labels]
        batch_hardness_labels = batch_hardness_labels.cuda()

    return encoder_input_ids, encoder_input_attention_mask, \
               batch_aligned_column_labels, batch_aligned_table_labels, \
               batch_aligned_question_ids, batch_aligned_column_info_ids, \
               batch_aligned_table_name_ids, batch_column_number_in_each_table, \
               batch_hardness_labels

# ----------------------(Jarven) For relation-across classification(Relation-Resd) -------------------
def prepare_dataset_inputs_and_labels(dataset, tokenizer):
    '''
    Input is a dataset(Train or Dev) and a tokenizer
    step1 : Take all information into corresponding list, respectively
    step2 : Get question, table_name, column_name_in_the_table
    step3 : Get train_input "question|t1:c11,c12,...|t2:c21,c22,...|...|tx:cx1,cx2,..."
    step4 : Use Tokenizer to obtain train_input_ids
    '''
    dataset_length = len(dataset)

    # batch_questions = [data['question'] for data in dataset]
    batch_questions = [data[0] for data in dataset]

    # batch_table_names = [data['all_table_names'] for data in dataset]
    # batch_table_labels = [data['all_table_labels'] for data in dataset]
    batch_table_names = [data[1] for data in dataset]
    batch_table_labels = [data[2] for data in dataset]

    # batch_column_infos = [data['all_column_infos'] for data in dataset]
    # batch_column_labels = [data['all_column_labels'] for data in dataset]
    batch_column_infos = [data[3] for data in dataset]
    batch_column_labels = [data[4] for data in dataset]

    batch_db_id = [data[5] for data in dataset] # capture db_id

    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
    train_input, train_input_ids = [], []
    for data_id in range(dataset_length):
        input_tokens = [batch_questions[data_id]]
        table_names_in_one_db = batch_table_names[data_id]
        column_infos_in_one_db = batch_column_infos[data_id]
        db_id_in_one_db= batch_db_id[data_id][0]

        batch_column_number_in_each_table.append(
            [len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []
        # --------- add db_id-----------
        input_tokens.append("|")
        input_tokens.append(db_id_in_one_db)
        # ------------------------------
        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")

            for column_info in column_infos_in_one_db[table_id]:
                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")

            input_tokens = input_tokens[:-1]

        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(       # this is what I want???
        batch_input_tokens,
        # return_tensors='pt',  # ***
        is_split_into_words=True,
        padding=False,      # ***
        max_length=512,
        truncation=True
    )

    # batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    # batch_aligned_table_labels, batch_aligned_column_labels = [], []

    # # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    # for data_id in range(dataset_length):
    #     word_ids = tokenized_inputs.word_ids(batch_index=data_id)
    #
    #     aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
    #     aligned_table_labels, aligned_column_labels = [], []
    #
    #     # align question tokens
    #     for token_id, word_id in enumerate(word_ids):
    #         if word_id == 0:
    #             aligned_question_ids.append(token_id)
    #
    #     # align table names
    #     for t_id, table_name_id in enumerate(batch_table_name_ids[data_id]):
    #         temp_list = []
    #         for token_id, word_id in enumerate(word_ids):
    #             if table_name_id == word_id:
    #                 temp_list.append(token_id)
    #         # if the tokenizer doesn't discard current table name
    #         if len(temp_list) != 0:
    #             aligned_table_name_ids.append(temp_list)
    #             aligned_table_labels.append(batch_table_labels[data_id][t_id])
    #
    #     # align column names
    #     for c_id, column_id in enumerate(batch_column_info_ids[data_id]):
    #         temp_list = []
    #         for token_id, word_id in enumerate(word_ids):
    #             if column_id == word_id:
    #                 temp_list.append(token_id)
    #         # if the tokenizer doesn't discard current column name
    #         if len(temp_list) != 0:
    #             aligned_column_info_ids.append(temp_list)
    #             aligned_column_labels.append(batch_column_labels[data_id][c_id])
    #
    #     batch_aligned_question_ids.append(aligned_question_ids)
    #     batch_aligned_table_name_ids.append(aligned_table_name_ids)
    #     batch_aligned_column_info_ids.append(aligned_column_info_ids)
    #     batch_aligned_table_labels.append(aligned_table_labels)
    #     batch_aligned_column_labels.append(aligned_column_labels)

    # # update column number in each table (because some tables and columns are discarded)
    # for data_id in range(dataset_length):
    #     if len(batch_column_number_in_each_table[data_id]) > len(batch_aligned_table_labels[data_id]):
    #         batch_column_number_in_each_table[data_id] = batch_column_number_in_each_table[data_id][
    #                                                       : len(batch_aligned_table_labels[data_id])]
    #
    #     if sum(batch_column_number_in_each_table[data_id]) > len(batch_aligned_column_labels[data_id]):
    #         truncated_column_number = sum(batch_column_number_in_each_table[data_id]) - len(
    #             batch_aligned_column_labels[data_id])
    #         batch_column_number_in_each_table[data_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]       # train_input_ids
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    # batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]    # the target of classifier
    # batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

    # if torch.cuda.is_available():
    #     encoder_input_ids = encoder_input_ids.cuda()
    #     encoder_input_attention_mask = encoder_input_attention_mask.cuda()
    #     # batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
    #     # batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    return encoder_input_ids, encoder_input_attention_mask#, \
           # batch_aligned_column_labels, batch_aligned_table_labels, \
           # batch_aligned_question_ids, batch_aligned_column_info_ids, \
           # batch_aligned_table_name_ids, batch_column_number_in_each_table

def get_dataset_relations(train_dataset, tokenizer, mode = "train", edge_type = 'Default'):
    # step1 目标：获取train_input_ids
    train_input_ids, train_input_attention_mask = prepare_dataset_inputs_and_labels(train_dataset, tokenizer)

    # train_input_ids = train_input_ids.tolist()        # tensor 2 list
    # step2 目标：获取relation matrix
    relation_matrix_l = preprocess_by_dataset(  # *** RASAT is Here !!!
        './dataset_files/',
        'spider',
        train_input_ids,
        mode,
        edge_type=edge_type,    #'Default'
        use_coref=False,
        use_dependency=True
    )
    # step3 目标：Put relation into datatset
    # def add_relation_info_train(example, idx, relation_matrix_l=relation_matrix_l):
    #     example['relations'] = relation_matrix_l[idx]
    #     return example
    # train_dataset = train_dataset.map(add_relation_info_train, with_indices=True)
    # for index in range(len(train_dataset[0])):
    #     train_dataset['relation'][index] = relation_matrix_l[index]

    return relation_matrix_l
# --------------------------------------------------

def _train(opt):
    print(opt)
    set_seed(opt.seed)

    patience = opt.patience if opt.patience > 0 else float('inf')

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    # tokens
    if opt.use_rasat is not True:
        if 'roberta' or 'RoBerta' in opt.model_name_or_path:
            tokenizer = RobertaTokenizerFast.from_pretrained(
                opt.model_name_or_path,
                add_prefix_space=True
            )
        elif 'bert' in opt.model_name_or_path:
            tokenizer = BertTokenizerFast.from_pretrained(
                opt.model_name_or_path,
                add_prefix_space=True
            )
        elif 't5' in opt.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                opt.model_name_or_path,  # 't5-',
                cache_dir='./transformers_cache',
                use_fast=True,
                revision='main',
                use_auth_token=None
            )
        tokenizer.add_tokens(AddedToken("[FK]"))

        if opt.resd_preprocessed_data is True:
            train_dataset = HardnessClassifierDatasetForRESD(dir_= opt.train_filepath)
            train_dataloder = DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                collate_fn=lambda x: x
            )
            dev_dataset = HardnessClassifierDatasetForRESD(dir_=opt.dev_filepath)
            dev_dataloder = DataLoader(
                dev_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                collate_fn=lambda x: x
            )
        # Train dataset
        else:
            train_dataset = HardnessClassifierDataset(
                dir_=opt.train_filepath,  # opt.dev_filepath*****!!!!! Just for speed
                use_contents=opt.use_contents,
                add_fk_info=opt.add_fk_info
            )
            train_dataloder = DataLoader(
                train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                collate_fn=lambda x: x
            )
            dev_dataset = HardnessClassifierDataset(
                dir_=opt.dev_filepath,
                use_contents=opt.use_contents,
                add_fk_info=opt.add_fk_info
            )
            dev_dataloder = DataLoader(
                dev_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                collate_fn=lambda x: x
            )
    # else:
    #     # -------------------T5-relation Tokenizer(Jarven)-------------------
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         opt.model_name_or_path,  # 't5-small',
    #         cache_dir='./transformers_cache',
    #         use_fast=True,
    #         revision='main',
    #         use_auth_token=None
    #     )
    #     assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    #     if isinstance(tokenizer, T5TokenizerFast):
    #         # In T5 `<` is OOV, see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/restore_oov.py
    #         tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    #     tokenizer.add_tokens(AddedToken("[FK]"))
    #     # prepare for relation
    #     train_dataset_relation = ColumnAndTableRelationDataset(
    #         dir_=opt.train_filepath,
    #         use_contents=opt.use_contents,
    #         add_fk_info=opt.add_fk_info
    #     )
    #     # -----------------Add Relations into TrainDataset (Jarven)-----------------
    #     assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are currently supported"
    #     train_dataset_with_relation = get_dataset_relations(train_dataset_relation, tokenizer, mode="train", edge_type=opt.edge_type)  # *****!!!!! Just for speed
    #
    #     # Train dataset Load
    #     train_dataset = ColumnAndTableClassifierDataset(
    #         dir_=opt.train_filepath,  #  opt.dev_filepath*****!!!!! Just for speed
    #         use_contents=opt.use_contents,
    #         add_fk_info=opt.add_fk_info,
    #         relations=train_dataset_with_relation    # relation matrix
    #     )
    #     train_dataloder = DataLoader(
    #         train_dataset,
    #         batch_size=opt.batch_size,
    #         shuffle=True,
    #         collate_fn=lambda x: x
    #     )
    #
    #     # Dev dataset Load
    #     dev_dataset_relation = ColumnAndTableRelationDataset(
    #         dir_=opt.dev_filepath,
    #         use_contents=opt.use_contents,
    #         add_fk_info=opt.add_fk_info
    #     )
    #
    #     dev_dataset_with_relation = get_dataset_relations(dev_dataset_relation, tokenizer, mode="dev", edge_type=opt.edge_type)
    #
    #     dev_dataset = ColumnAndTableClassifierDataset(
    #         dir_=opt.dev_filepath,
    #         use_contents=opt.use_contents,
    #         add_fk_info=opt.add_fk_info,
    #         relations=dev_dataset_with_relation
    #     )
    #
    #     dev_dataloder = DataLoader(
    #         dev_dataset,
    #         batch_size=opt.batch_size,
    #         shuffle=False,
    #         collate_fn=lambda x: x
    #     )
    #     # ------------------------------------------------------------

    # # Train dataset Load
    # train_dataset = ColumnAndTableClassifierDataset(
    #     dir_=opt.train_filepath,
    #     use_contents=opt.use_contents,
    #     add_fk_info=opt.add_fk_info
    # )

    # train_dataloder = DataLoader(
    #     train_dataset,
    #     batch_size = opt.batch_size,
    #     shuffle = True,
    #     collate_fn = lambda x: x
    # )



    # ---------------------- Initial Model (Jarven)--------------------
    if 't5' in opt.model_name_or_path:
        config = AutoConfig.from_pretrained(
            opt.model_name_or_path,
            cache_dir='./transformers_cache',
            revision='main',
            use_auth_token=None,
            max_length=512,
            num_beams=4,
            num_beam_groups=1,
            diversity_penalty=None,
            gradient_checkpointing=True,
            use_cache=not True,
        )
        if opt.use_rasat:
            _, _, num_relations = get_relation2id_dict('Default', use_coref=False,
                                                       use_dependency=True)
            config.num_relations = num_relations
            print("===================================================")
            print("Num of relations uesd in RASAT is : ", num_relations)
            print("===================================================")
            print("Use relation model.")
        else:    config.num_relations = None

        model = JarvenClassifier(
            config=config,
            model_name_or_path=opt.model_name_or_path,
            vocab_size=len(tokenizer),
            mode=opt.mode
        )
    elif 'roberta' or 'RoBerta' in opt.model_name_or_path:
        model = HardnessClassifier(
            model_name_or_path=opt.model_name_or_path,
            vocab_size=len(tokenizer),
            mode=opt.mode
        )
    # ------------------------------------------------------------------

    # initialize model
    # model = MyClassifier(
    #     model_name_or_path = opt.model_name_or_path,
    #     vocab_size = len(tokenizer),
    #     mode = opt.mode
    # )

    if torch.cuda.is_available():
        model = model.cuda()

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # evaluate model for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider)
    # num_checkpoint_steps = int(1.42857*len(train_dataset)/opt.batch_size)   # 1250
    num_checkpoint_steps = 3000
    optimizer = optim.AdamW(
        params = model.parameters(), 
        lr = opt.learning_rate
    )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    best_score, best_score_epoch, early_stop_step, train_step = 0, 0, 0, 0
    # encoder_loss_func = ClassifierLoss(alpha = opt.alpha, gamma = opt.gamma)
    encoder_loss_func = HardnessClassifierLoss(alpha = opt.alpha, gamma = opt.gamma)
    
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        for batch in tqdm(train_dataloder):
            model.train()
            train_step += 1
            # if train_step % 1000 == 0:
                # print(f"train step: {train_step}.")
            if opt.use_rasat and 't5' in opt.model_name_or_path:
                encoder_input_ids, encoder_input_attention_mask, \
                batch_column_labels, batch_table_labels, \
                batch_aligned_question_ids, \
                batch_aligned_column_info_ids, batch_aligned_table_name_ids, batch_column_number_in_each_table, \
                batch_hardness_labels, batch_relations = prepare_batch_inputs_and_labels(batch, tokenizer, opt.use_rasat)  # ***

                model_outputs = model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table,
                    batch_relations=batch_relations
                )
            elif opt.resd_preprocessed_data is True:
                'When using RESD preprocessd dataset for Training'
                encoder_input_ids, encoder_input_attention_mask, \
                batch_column_labels, batch_table_labels, \
                batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                batch_column_number_in_each_table, batch_hardness_labels = prepare_RESDdataset_input_and_labels(batch, tokenizer)

                model_outputs = model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table,
                )
            else:
                encoder_input_ids, encoder_input_attention_mask, \
                batch_column_labels, batch_table_labels, \
                batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                batch_column_number_in_each_table, batch_hardness_labels = prepare_batch_inputs_and_labels(batch, tokenizer, opt.use_rasat)  # ***

                model_outputs = model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table,
                )

            loss = encoder_loss_func.compute_loss(model_outputs["hardness"], batch_hardness_labels)

            loss.backward()

            # update lr
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
            
            if train_step % num_checkpoint_steps == 0:
                print(f"At {train_step} training step, start an evaluation.")
                model.eval()

                hardness_labels_for_auc = []
                hardness_pred_probs_for_auc = []


                for batch in dev_dataloder:
                    if opt.use_rasat and 't5' in opt.model_name_or_path:
                        encoder_input_ids, encoder_input_attention_mask, \
                        batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
                        batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                        batch_column_number_in_each_table, \
                        batch_relations = prepare_batch_inputs_and_labels(batch, tokenizer, opt.use_rasat)

                        with torch.no_grad():
                            model_outputs = model(
                                encoder_input_ids,
                                encoder_input_attention_mask,
                                batch_aligned_question_ids,
                                batch_aligned_column_info_ids,
                                batch_aligned_table_name_ids,
                                batch_column_number_in_each_table,
                                batch_relations=batch_relations
                            )
                    elif opt.resd_preprocessed_data is True:
                        'When using RESD preprocessd dataset for Dev'
                        encoder_input_ids, encoder_input_attention_mask, \
                        batch_column_labels, batch_table_labels, \
                        batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                        batch_column_number_in_each_table, batch_hardness_labels = prepare_RESDdataset_input_and_labels(
                            batch, tokenizer)

                        with torch.no_grad():
                            model_outputs = model(
                                encoder_input_ids,
                                encoder_input_attention_mask,
                                batch_aligned_question_ids,
                                batch_aligned_column_info_ids,
                                batch_aligned_table_name_ids,
                                batch_column_number_in_each_table,
                                batch_relations=None
                            )
                    else:
                        encoder_input_ids, encoder_input_attention_mask, \
                        batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
                        batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
                        batch_column_number_in_each_table, batch_hardness_labels  = prepare_batch_inputs_and_labels(batch, tokenizer, opt.use_rasat)

                        with torch.no_grad():
                            model_outputs = model(
                                encoder_input_ids,
                                encoder_input_attention_mask,
                                batch_aligned_question_ids,
                                batch_aligned_column_info_ids,
                                batch_aligned_table_name_ids,
                                batch_column_number_in_each_table,
                                batch_relations=None
                            )

                    for batch_id, hardness_logits in enumerate(model_outputs["hardness"]):
                        hardness_pred_probs = torch.argmax(torch.nn.functional.softmax(hardness_logits, dim=-1)) # ***
                        hardness_pred_probs_for_auc.append(hardness_pred_probs.cpu().tolist())
                    hardness_labels_for_auc.extend(batch_hardness_labels.cpu().tolist())

                    # for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
                    #     column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)
                    #
                    #     column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
                    #     column_labels_for_auc.extend(batch_column_labels[batch_id].cpu().tolist())

                # calculate AUC score for table classification   cls_metric
                hardness_auc = cls_metric(hardness_labels_for_auc, hardness_pred_probs_for_auc)
                # calculate AUC score for column classification
                # column_auc = cls_metric(column_labels_for_auc, column_pred_probs_for_auc)
                # print("hardness CLS:", hardness_auc)
                list_key, list_value = [], []
                for key, value in hardness_auc.items():
                    list_key.append(key)
                    list_value.append(value)

                toral_auc_score = hardness_auc
                # Record into writer
                if writer is not None:
                    # Total Acc
                    writer.add_scalar('Hardness Total Acc', toral_auc_score['accuracy'], train_step/num_checkpoint_steps)
                    # Precision
                    writer.add_scalar('Easy Precision', toral_auc_score['easy']['precision'], train_step/num_checkpoint_steps)
                    writer.add_scalar('Medium Precision', toral_auc_score['medium']['precision'], train_step / num_checkpoint_steps)
                    writer.add_scalar('Hard Precision', toral_auc_score['hard']['precision'], train_step / num_checkpoint_steps)
                    writer.add_scalar('Extra-hard Precision', toral_auc_score['extra-hard']['precision'], train_step / num_checkpoint_steps)
                    # Recall
                    writer.add_scalar('Easy Recall', toral_auc_score['easy']['recall'],train_step / num_checkpoint_steps)
                    writer.add_scalar('Medium Recall', toral_auc_score['medium']['recall'],train_step / num_checkpoint_steps)
                    writer.add_scalar('Hard Recall', toral_auc_score['hard']['recall'],train_step / num_checkpoint_steps)
                    writer.add_scalar('Extra-hard Recall', toral_auc_score['extra-hard']['recall'],train_step / num_checkpoint_steps)
                    # Best Acc
                    writer.add_scalar('Best Acc', best_score, train_step / num_checkpoint_steps)

                print("Precision:")
                print("Easy:\t\tPrecision: ", toral_auc_score['easy']['precision'], "\tRecall: ",\
                      toral_auc_score['easy']['recall'], "\tF1:", toral_auc_score['easy']['f1-score'], "\tnumber:",
                      toral_auc_score['easy']['support'])
                print("Medium:\t\tPrecision: ", toral_auc_score['medium']['precision'], "\tRecall: ", \
                      toral_auc_score['medium']['recall'], "\tF1:", toral_auc_score['medium']['f1-score'], "\tnumber:",
                      toral_auc_score['medium']['support'])
                print("Hard:\t\tPrecision: ", toral_auc_score['hard']['precision'], "\tRecall: ", \
                      toral_auc_score['hard']['recall'], "\tF1:", toral_auc_score['hard']['f1-score'], "\tnumber:",
                      toral_auc_score['hard']['support'])
                print("Extra-Hard:\tPrecision: ", toral_auc_score['extra-hard']['precision'], "\tRecall: ", \
                      toral_auc_score['extra-hard']['recall'], "\tF1:", toral_auc_score['extra-hard']['f1-score'], "\tnumber:",
                      toral_auc_score['extra-hard']['support'])
                # save the best ckpt
                if toral_auc_score['accuracy'] >= best_score:
                    best_score = toral_auc_score['accuracy']
                    best_score_epoch = epoch
                    # save model
                    os.makedirs(opt.save_path, exist_ok = True)
                    torch.save(model.state_dict(), opt.save_path + "/dense_classifier.pt")
                    model.plm_encoder.config.save_pretrained(save_directory = opt.save_path)
                    tokenizer.save_pretrained(save_directory = opt.save_path)
                print("Current Accuracy", toral_auc_score['accuracy'], "Best Accurcy:", best_score,
                      "and its epoch:", best_score_epoch)
        #             early_stop_step = 0
        #         else:
        #             early_stop_step += 1
        #
        #         print("early_stop_step:", early_stop_step)
        #
        #     if early_stop_step >= patience:
        #         break
        #
        # if early_stop_step >= patience:
        #     print("Classifier training process triggers early stopping.")
        #     break
    
    print("best auc score:", best_score)

def _test(opt):
    set_seed(opt.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device

    # load tokenizer
    if 'roberta' in opt.model_name_or_path:
        tokenizer = RobertaTokenizerFast.from_pretrained(
            opt.save_path,
            add_prefix_space = True
        )
    elif 'bert' in opt.model_name_or_path:
        tokenizer = BertTokenizerFast.from_pretrained(
            opt.model_name_or_path,
            add_prefix_space=True
        )
    # -------------------------
    # Test dataset
    if opt.resd_preprocessed_data is True:
        dataset = HardnessClassifierDatasetForRESD(dir_=opt.dev_filepath)
        dataloder = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )
    else:
        dataset = HardnessClassifierDataset(
            dir_=opt.dev_filepath,
            use_contents=opt.use_contents,
            add_fk_info=opt.add_fk_info
        )
        dataloder = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )
    # --------------------------
    # initialize model
    model = HardnessClassifier(
        model_name_or_path=opt.save_path,
        vocab_size=len(tokenizer),
        mode=opt.mode
    )

    # load fine-tuned params
    model.load_state_dict(torch.load(opt.save_path + "/dense_classifier.pt", map_location=torch.device('cuda')))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # table_labels_for_auc, column_labels_for_auc = [], []
    # table_pred_probs_for_auc, column_pred_probs_for_auc = [], []
    hardness_labels_for_auc = []
    hardness_pred_probs_for_auc = []

    returned_table_pred_probs, returned_column_pred_probs = [], []

    for batch in tqdm(dataloder):
        # encoder_input_ids, encoder_input_attention_mask, \
        #     batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
        #     batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
        #     batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, tokenizer)

        # -------------

        if opt.resd_preprocessed_data is True:
            'When using RESD preprocessd dataset for Dev'
            encoder_input_ids, encoder_input_attention_mask, \
            batch_column_labels, batch_table_labels, \
            batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
            batch_column_number_in_each_table, batch_hardness_labels = prepare_RESDdataset_input_and_labels(
                batch, tokenizer)

            with torch.no_grad():
                model_outputs = model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table,
                    batch_relations=None
                )

        else:
            encoder_input_ids, encoder_input_attention_mask, \
            batch_column_labels, batch_table_labels, \
            batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
            batch_column_number_in_each_table, batch_hardness_labels = prepare_batch_inputs_and_labels(batch, tokenizer,
                                                                                                       opt.use_rasat)  # ***
            # -------------
            with torch.no_grad():
                model_outputs = model(
                    encoder_input_ids,
                    encoder_input_attention_mask,
                    batch_aligned_question_ids,
                    batch_aligned_column_info_ids,
                    batch_aligned_table_name_ids,
                    batch_column_number_in_each_table
                )

        # for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
        #     table_pred_probs = torch.nn.functional.softmax(table_logits, dim = 1)
        #     returned_table_pred_probs.append(table_pred_probs[:, 1].cpu().tolist())
        #
        #     table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())
        #     table_labels_for_auc.extend(batch_table_labels[batch_id].cpu().tolist())
        #
        # for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
        #     column_number_in_each_table = batch_column_number_in_each_table[batch_id]
        #     column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)
        #     returned_column_pred_probs.append([column_pred_probs[:, 1].cpu().tolist()[sum(column_number_in_each_table[:table_id]):sum(column_number_in_each_table[:table_id+1])] \
        #         for table_id in range(len(column_number_in_each_table))])
        #
        #     column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
        #     column_labels_for_auc.extend(batch_column_labels[batch_id].cpu().tolist())
        # ---
        for batch_id, hardness_logits in enumerate(model_outputs["hardness"]):
            hardness_pred_probs = torch.argmax(torch.nn.functional.softmax(hardness_logits, dim=-1))  # ***
            hardness_pred_probs_for_auc.append(hardness_pred_probs.cpu().tolist())
        hardness_labels_for_auc.extend(batch_hardness_labels.cpu().tolist())
        # ---

    if opt.mode == "eval":
        # calculate AUC score for table classification   cls_metric
        hardness_auc = cls_metric(hardness_labels_for_auc, hardness_pred_probs_for_auc)
        toral_auc_score = hardness_auc
        print("Precision:")
        print("Easy:\t\tPrecision: ", toral_auc_score['easy']['precision'], "\tRecall: ", \
              toral_auc_score['easy']['recall'], "\tF1:", toral_auc_score['easy']['f1-score'], "\tnumber:",
              toral_auc_score['easy']['support'])
        print("Medium:\t\tPrecision: ", toral_auc_score['medium']['precision'], "\tRecall: ", \
              toral_auc_score['medium']['recall'], "\tF1:", toral_auc_score['medium']['f1-score'], "\tnumber:",
              toral_auc_score['medium']['support'])
        print("Hard:\t\tPrecision: ", toral_auc_score['hard']['precision'], "\tRecall: ", \
              toral_auc_score['hard']['recall'], "\tF1:", toral_auc_score['hard']['f1-score'], "\tnumber:",
              toral_auc_score['hard']['support'])
        print("Extra-Hard:\tPrecision: ", toral_auc_score['extra-hard']['precision'], "\tRecall: ", \
              toral_auc_score['extra-hard']['recall'], "\tF1:", toral_auc_score['extra-hard']['f1-score'], "\tnumber:",
              toral_auc_score['extra-hard']['support'])
        print("Current Accuracy", toral_auc_score['accuracy'])
    
    return hardness_pred_probs_for_auc

if __name__ == "__main__":
    opt = parse_option()
    if opt.mode == "train":
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        hardness_pred_probs_for_auc = _test(opt)
        
        with open(opt.dev_filepath, "r") as f:
            dataset = json.load(f)
        
        # record predicted probability
        truncated_data_info = []
        for data_id, data in enumerate(dataset):
            # table_num = len(data["hardness_labels"])
            hardness_pred_probs = hardness_pred_probs_for_auc[data_id]

            # truncated_table_ids = []
            # column_pred_probs = []
            # for table_id in range(table_num):
            #     if table_id >= len(total_column_pred_probs[data_id]):
            #         truncated_table_ids.append(table_id)
            #         column_pred_probs.append([-1 for _ in range(len(data["column_labels"][table_id]))])
            #         continue
            #     if len(total_column_pred_probs[data_id][table_id]) == len(data["column_labels"][table_id]):
            #         column_pred_probs.append(total_column_pred_probs[data_id][table_id])
            #     else:
            #         truncated_table_ids.append(table_id)
            #         truncated_column_num = len(data["column_labels"][table_id]) - len(total_column_pred_probs[data_id][table_id])
            #         column_pred_probs.append(total_column_pred_probs[data_id][table_id] + [-1 for _ in range(truncated_column_num)])
            
            # data["column_pred_probs"] = column_pred_probs
            # data["table_pred_probs"] = table_pred_probs
            data["hardness_pred_probs"] = hardness_pred_probs
            if data["hardness_pred_probs"] == 0:    Hds = "[/easy] "
            elif data["hardness_pred_probs"] == 1:  Hds = "[/medium] "
            elif data["hardness_pred_probs"] == 2:  Hds = "[/hard] "
            elif data["hardness_pred_probs"] == 2:  Hds = "[/extra] "

            data["input_sequence"] = Hds + data["input_sequence"]

        with open(opt.output_filepath, "w") as f:
            # f.write(json.dumps(dataset, indent = 2))
            json_str = json.dumps(dataset, indent=2)
            f.write(json_str)
            print("Recording Test dataset have Finished !")