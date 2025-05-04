import torch
import torch.nn as nn

from transformers import AutoConfig, RobertaModel, BertModel
# ---------------------added By Jarven ----------------------------------
from seq2seq.model.t5_relation_model import T5ForConditionalGeneration as T5_Relation
from seq2seq.model.t5_original_model import T5ForConditionalGeneration as T5_Original
from transformers import T5ForConditionalGeneration as T5_Pretrained
# ----------------------------------------------------------------------

class MyClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        vocab_size,
        mode
    ):
        super(MyClassifier, self).__init__()

        if mode in ["eval", "test"]:
            # load config
            config = AutoConfig.from_pretrained(model_name_or_path)
            # randomly initialize model's parameters according to the config
            self.plm_encoder = RobertaModel(config)
        elif mode == "train":
            self.plm_encoder = RobertaModel.from_pretrained(model_name_or_path)
            self.plm_encoder.resize_token_embeddings(vocab_size)

        else:
            raise ValueError()

        # ---------------- Jarven #
        if model_name_or_path=="roberta-base":
            input_dim = 768
        else:
            input_dim = 1024
        # ----------------  #

        # column cls head
        self.column_info_cls_head_linear1 = nn.Linear(1024, 256)
        self.column_info_cls_head_linear2 = nn.Linear(256, 2)
        
        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = 512,
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )

        # linear layer after column bi-lstm layer
        self.column_info_linear_after_pooling = nn.Linear(1024, 1024)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(1024, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)
        
        # table bi-lstm pooling layer

        self.table_name_bilstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = 512,
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )
        # linear layer after table bi-lstm layer
        self.table_name_linear_after_pooling = nn.Linear(1024, 1024)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # table-column cross-attention layer
        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim = 1024, num_heads = 8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p = 0.2)
    
    def table_column_cross_attention(
        self,
        table_name_embeddings_in_one_db, 
        column_info_embeddings_in_one_db, 
        column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                sum(column_number_in_each_table[:table_id]) : sum(column_number_in_each_table[:table_id+1]), :]
            
            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)
        
        # residual connection
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list, dim = 0)
        # row-wise L2 norm
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db

    def table_column_cls(
        self,
        # ---------- Input to plm -----------
        encoder_input_ids,
        encoder_input_attention_mask,
        # -----------------------------------
        batch_aligned_question_ids, # useless
        # ----------- Be index of table and column ---------
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table
    ):
        batch_size = encoder_input_ids.shape[0]
        
        encoder_output = self.plm_encoder(
            input_ids = encoder_input_ids,
            attention_mask = encoder_input_attention_mask,
            return_dict = True
        ) # encoder_output["last_hidden_state"].shape = (batch_size x seq_length x hidden_size)

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        # handle each data in current batch
        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :] # (seq_length x hidden_size)
            
            # obtain the embeddings of tokens in the question
            question_token_embeddings = sequence_embeddings[batch_aligned_question_ids[batch_id], :]

            # obtain table ids for each table
            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            # obtain column ids for each column
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            # obtain table embedding via bi-lstm pooling + a non-linear layer
            for table_name_ids in aligned_table_name_ids:
                table_name_embeddings = sequence_embeddings[table_name_ids, :]
                
                # BiLSTM pooling
                output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
                table_name_embedding = hidden_state_t[-2:, :].view(1, 1024)
                table_name_embedding_list.append(table_name_embedding)
            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim = 0)
            # non-linear mlp layer
            table_name_embeddings_in_one_db = self.leakyrelu(self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
            
            # obtain column embedding via bi-lstm pooling + a non-linear layer
            for column_info_ids in aligned_column_info_ids:
                column_info_embeddings = sequence_embeddings[column_info_ids, :]
                
                # BiLSTM pooling
                output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
                column_info_embedding = hidden_state_c[-2:, :].view(1, 1024)
                column_info_embedding_list.append(column_info_embedding)
            column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim = 0)
            # non-linear mlp layer
            column_info_embeddings_in_one_db = self.leakyrelu(self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))

            # table-column (tc) cross-attention
            table_name_embeddings_in_one_db = self.table_column_cross_attention(
                table_name_embeddings_in_one_db, 
                column_info_embeddings_in_one_db, 
                column_number_in_each_table
            )
            
            # calculate table 0-1 logits
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)

            # calculate column 0-1 logits
            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        batch_aligned_question_ids,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table,
            batch_relations=None
    ):  
        batch_table_name_cls_logits, batch_column_info_cls_logits \
            = self.table_column_cls(
            encoder_input_ids,
            encoder_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table
        )

        return {
            "batch_table_name_cls_logits" : batch_table_name_cls_logits, 
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }


class JarvenClassifier(nn.Module):
    def __init__(
            self,
            config,
            model_name_or_path,
            vocab_size,
            mode
    ):
        super(JarvenClassifier, self).__init__()

        self.model_name_or_path = model_name_or_path

        if mode in ["eval", "test"]:
            # load config
            # config = AutoConfig.from_pretrained(model_name_or_path)
            # randomly initialize model's parameters according to the config
            # self.plm_encoder = RobertaModel(config)
            self.plm_encoder = T5_Relation(config)

        elif mode == "train":
            # self.plm_encoder = RobertaModel.from_pretrained(model_name_or_path)
            # self.plm_encoder.resize_token_embeddings(vocab_size)
            config.is_encoder_decoder = False
            self.plm_encoder = T5_Relation(config)

            model_pretrained = T5_Pretrained.from_pretrained(model_name_or_path)
            parameter_dict = model_pretrained.state_dict()
            model_dict = self.plm_encoder.state_dict()
            model_dict.update(parameter_dict)
            self.plm_encoder.load_state_dict(model_dict)

            self.plm_encoder.resize_token_embeddings(vocab_size)

        else:
            raise ValueError()

        # ---------------- Jarven #
        if self.model_name_or_path == "roberta-base":
            input_dim = 768
        elif self.model_name_or_path == "roberta-large":
            input_dim = 1024
        elif self.model_name_or_path == "t5-small":
            input_dim = 512
        elif self.model_name_or_path == "t5-base":
            input_dim = 768
        elif self.model_name_or_path == "t5-large":
            input_dim = 1024
        # ----------------- #
        # column cls head

        self.column_info_cls_head_linear1 = nn.Linear(1024, 256)
        self.column_info_cls_head_linear2 = nn.Linear(256, 2)

        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )

        # linear layer after column bi-lstm layer
        self.column_info_linear_after_pooling = nn.Linear(1024, 1024)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(1024, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)

        # table bi-lstm pooling layer
        self.table_name_bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=512,
            num_layers=2,
            dropout=0,
            bidirectional=True
        )
        # linear layer after table bi-lstm layer
        self.table_name_linear_after_pooling = nn.Linear(1024, 1024)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # table-column cross-attention layer
        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p=0.2)

    def table_column_cross_attention(
            self,
            table_name_embeddings_in_one_db,
            column_info_embeddings_in_one_db,
            column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                                                  sum(column_number_in_each_table[:table_id]): sum(
                                                      column_number_in_each_table[:table_id + 1]), :]

            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)

        # residual connection
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list,
                                                                                      dim=0)
        # row-wise L2 norm
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db

    def table_column_cls(
            self,
            encoder_input_ids,
            encoder_input_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table,
            batch_relations=None
            # ---Jarven  We need to find the API of relations, and put relation matrix into model.
    ):
        batch_size = encoder_input_ids.shape[0]

        encoder_output = self.plm_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            return_dict=True,
            # --------fake---------- #
            decoder_input_ids=encoder_input_ids,
            decoder_attention_mask=encoder_input_attention_mask,
            # ---------***realtion***--------- #
            relations = batch_relations
        )  # encoder_output["last_hidden_state"].shape = (batch_size x seq_length x hidden_size)

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        # handle each data in current batch
        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            try:
                sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :]  # (seq_length x hidden_size)
            except:
                sequence_embeddings = encoder_output["encoder_last_hidden_state"][batch_id, :, :]  # (seq_length x hidden_size)

            # obtain the embeddings of tokens in the question
            question_token_embeddings = sequence_embeddings[batch_aligned_question_ids[batch_id], :]

            # obtain table ids for each table
            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            # obtain column ids for each column
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            # obtain table embedding via bi-lstm pooling + a non-linear layer
            for table_name_ids in aligned_table_name_ids:
                table_name_embeddings = sequence_embeddings[table_name_ids, :]

                # BiLSTM pooling
                output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
                table_name_embedding = hidden_state_t[-2:, :].view(1, 1024)
                table_name_embedding_list.append(table_name_embedding)
            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim=0)
            # non-linear mlp layer
            table_name_embeddings_in_one_db = self.leakyrelu(
                self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))

            # obtain column embedding via bi-lstm pooling + a non-linear layer
            for column_info_ids in aligned_column_info_ids:
                column_info_embeddings = sequence_embeddings[column_info_ids, :]

                # BiLSTM pooling
                output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
                column_info_embedding = hidden_state_c[-2:, :].view(1, 1024)
                column_info_embedding_list.append(column_info_embedding)
            column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim=0)
            # non-linear mlp layer
            column_info_embeddings_in_one_db = self.leakyrelu(
                self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))

            # table-column (tc) cross-attention
            table_name_embeddings_in_one_db = self.table_column_cross_attention(
                table_name_embeddings_in_one_db,
                column_info_embeddings_in_one_db,
                column_number_in_each_table
            )

            # calculate table 0-1 logits
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)

            # calculate column 0-1 logits
            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(
            self,
            encoder_input_ids,
            encoder_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table,
            batch_relations = None    # ---Jarven
    ):
        batch_table_name_cls_logits, batch_column_info_cls_logits \
            = self.table_column_cls(
            encoder_input_ids,
            encoder_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table,
            batch_relations     # ---Jarven
        )

        return {
            "batch_table_name_cls_logits": batch_table_name_cls_logits,
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }


class HardnessClassifier(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            vocab_size,
            mode
    ):
        super(HardnessClassifier, self).__init__()

        if mode in ["eval", "test"]:
            # load config
            config = AutoConfig.from_pretrained(model_name_or_path)
            # randomly initialize model's parameters according to the config
            # if 'roberta' in model_name_or_path:
            self.plm_encoder = RobertaModel(config)
            print("model is RobertaModel")
            # else:
            #     self.plm_encoder = BertModel(config)
            #     print("model is BertModel")
        elif mode == "train":
            if 'roberta' or 'RoBerta' in model_name_or_path:
                self.plm_encoder = RobertaModel.from_pretrained(model_name_or_path)
                print("model is RobertaModel")
            else:
                self.plm_encoder = BertModel.from_pretrained(model_name_or_path)
                print("model is BertModel")
            self.plm_encoder.resize_token_embeddings(vocab_size)

        else:
            raise ValueError()

        # ---------------- Jarven ------------#
        if model_name_or_path == "roberta-base":
            input_dim = 768
        else:
            input_dim = 1024
        # ----------------Hardness------------#
        self.hardness_linear1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.hardness_linear2 = nn.Linear(256, 4)
        self.bn2 = nn.BatchNorm1d(4)
        self.hardness_linear3 = nn.Linear(4, 4)
        self.hardness_linear4 = nn.Linear(512*2, 256)
        self.hardness_linear5 = nn.Linear(256, 4)
        self.Norm = nn.BatchNorm1d(1, 2048)
        self.softmax_hardness = nn.Softmax()

        # -----------FewLinear-T-----------
        self.transposeLinear1 = nn.Linear(512, 128)
        self.bn_t1 = nn.BatchNorm1d(4, 128)
        self.transposeLinear2 = nn.Linear(128, 1)   # ***
        self.bn_t2 = nn.BatchNorm1d(4, 32)
        self.transposeLinear3 = nn.Linear(32, 1)
        # self.hardness_linear_pool = nn.Linear(4, 4)

        # column cls head
        # self.column_info_cls_head_linear1 = nn.Linear(1024, 256)
        # self.column_info_cls_head_linear2 = nn.Linear(256, 2)

        # self.bilstm = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=512,
        #     num_layers=2,
        #     dropout=0,
        #     bidirectional=True
        # )

        # column bi-lstm layer
        # self.column_info_bilstm = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=512,
        #     num_layers=2,
        #     dropout=0,
        #     bidirectional=True
        # )

        # linear layer after column bi-lstm layer
        # self.column_info_linear_after_pooling = nn.Linear(1024, 1024)

        # table cls head
        # self.table_name_cls_head_linear1 = nn.Linear(1024, 256)
        # self.table_name_cls_head_linear2 = nn.Linear(256, 2)

        # table bi-lstm pooling layer

        # self.table_name_bilstm = nn.LSTM(
        #     input_size=input_dim,
        #     hidden_size=512,
        #     num_layers=2,
        #     dropout=0,
        #     bidirectional=True
        # )
        # linear layer after table bi-lstm layer
        # self.table_name_linear_after_pooling = nn.Linear(1024, 1024)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        # table-column cross-attention layer
        # self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim=1024, num_heads=8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p=0.2)

    def table_column_cross_attention(
            self,
            table_name_embeddings_in_one_db,
            column_info_embeddings_in_one_db,
            column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                                                  sum(column_number_in_each_table[:table_id]): sum(
                                                      column_number_in_each_table[:table_id + 1]), :]

            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)

        # residual connection
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list,
                                                                                      dim=0)
        # row-wise L2 norm
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db

    def table_column_cls(
            self,
            # ---------- Input to plm -----------
            encoder_input_ids,
            encoder_input_attention_mask,
            # -----------------------------------
            # batch_aligned_question_ids,  # useless
            # # ----------- Be index of table and column ---------
            # batch_aligned_column_info_ids,
            # batch_aligned_table_name_ids,
            # batch_column_number_in_each_table
    ):
        batch_size = encoder_input_ids.shape[0]

        encoder_output = self.plm_encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            return_dict=True
        )  # encoder_output["last_hidden_state"].shape = (batch_size x seq_length x hidden_size)

        # batch_table_name_cls_logits, batch_column_info_cls_logits = [], []
        # batch_hardness_cls_logits = []

        # # handle each data in current batch
        # for batch_id in range(batch_size):
        #     column_number_in_each_table = batch_column_number_in_each_table[batch_id]
        #     sequence_embeddings = encoder_output["last_hidden_state"][batch_id, :, :]  # (seq_length x hidden_size)
        #     # ------------------Hardness -------------------
        #     hidden_state1 = self.dropout(self.leakyrelu(self.hardness_linear1(sequence_embeddings)))
        #     hidden_state2 = self.dropout(self.leakyrelu(self.hardness_linear2(hidden_state1)))
        #     hidden_state2_re = hidden_state2.view(1, 2048)
        #     hidden_state3 = self.leakyrelu(self.hardness_linear3(hidden_state2_re))
        #     hidden_state4 = self.leakyrelu(self.hardness_linear4(hidden_state3))
        #     output = self.hardness_linear5(hidden_state4)
        #     output_hardness_label = self.softmax_hardness(output)
        #     output_hardness_labels = torch.argmax(output_hardness_label, dim = 1)
        #     batch_hardness_cls_logits.append(output_hardness_labels)

            # output = self.softmax_hardness(hidden_state2)
            # output_hard_label = torch.argmax(output)
            # ------------------
        #     #
        #     # # obtain the embeddings of tokens in the question
        #     # question_token_embeddings = sequence_embeddings[batch_aligned_question_ids[batch_id], :]
        #     #
        #     # # obtain table ids for each table
        #     # aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
        #     # # obtain column ids for each column
        #     # aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]
        #     #
        #     # table_name_embedding_list, column_info_embedding_list = [], []
        #     #
        #     # # obtain table embedding via bi-lstm pooling + a non-linear layer
        #     # for table_name_ids in aligned_table_name_ids:
        #     #     table_name_embeddings = sequence_embeddings[table_name_ids, :]
        #     #
        #     #     # BiLSTM pooling
        #     #     output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
        #     #     table_name_embedding = hidden_state_t[-2:, :].view(1, 1024)
        #     #     table_name_embedding_list.append(table_name_embedding)
        #     # table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim=0)
        #     # # non-linear mlp layer
        #     # table_name_embeddings_in_one_db = self.leakyrelu(
        #     #     self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
        #     #
        #     # # obtain column embedding via bi-lstm pooling + a non-linear layer
        #     # for column_info_ids in aligned_column_info_ids:
        #     #     column_info_embeddings = sequence_embeddings[column_info_ids, :]
        #     #
        #     #     # BiLSTM pooling
        #     #     output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
        #     #     column_info_embedding = hidden_state_c[-2:, :].view(1, 1024)
        #     #     column_info_embedding_list.append(column_info_embedding)
        #     # column_info_embeddings_in_one_db = torch.cat(column_info_embedding_list, dim=0)
        #     # # non-linear mlp layer
        #     # column_info_embeddings_in_one_db = self.leakyrelu(
        #     #     self.column_info_linear_after_pooling(column_info_embeddings_in_one_db))
        #     #
        #     # # table-column (tc) cross-attention
        #     # table_name_embeddings_in_one_db = self.table_column_cross_attention(
        #     #     table_name_embeddings_in_one_db,
        #     #     column_info_embeddings_in_one_db,
        #     #     column_number_in_each_table
        #     # )
        #     #
        #     # # calculate table 0-1 logits
        #     # table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
        #     # table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
        #     # table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)
        #     #
        #     # # calculate column 0-1 logits
        #     # column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
        #     # column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
        #     # column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)
        #     #
        #     # batch_table_name_cls_logits.append(table_name_cls_logits)
        #     # batch_column_info_cls_logits.append(column_info_cls_logits)

        sequence_embeddings = encoder_output["last_hidden_state"]  # (seq_length x hidden_size)
        # ------------------Hardness -------------------
        # output_t, (hidden_state_t, cell_state_t) = self.bilstm(sequence_embeddings)
        # hidden_state1 = self.dropout(self.leakyrelu(self.hardness_linear1(output_t)))
        # hidden_state2 = self.leakyrelu(self.hardness_linear2(hidden_state1))
        # hidden_state2_re = hidden_state2.view(batch_size, 1, 1024*2)
        # hidden_state3 = self.leakyrelu(self.hardness_linear3(hidden_state2_re))
        # hidden_state4 = self.leakyrelu(self.hardness_linear4(hidden_state3))
        # batch_hardness_cls_logits = self.hardness_linear5(hidden_state4)
        # batch_hardness_cls_logits = batch_hardness_cls_logits.squeeze(1)
        # # batch_hardness_cls_logits = self.softmax_hardness(output)
        # # batch_hardness_cls_logits = torch.argmax(output_hardness_label, dim=2)

        # ----------------------Few Linear-------------------
        output = self.dropout(self.leakyrelu(self.hardness_linear1(sequence_embeddings)))
        hidden_state1 = self.dropout(self.leakyrelu(self.hardness_linear2(output)))

        hidden_state1_T = torch.transpose(hidden_state1, 1, 2)
        hidden_state2_T = self.leakyrelu(self.transposeLinear1(hidden_state1_T))
        hidden_state3_T = self.leakyrelu(self.transposeLinear2(hidden_state2_T))
        # hidden_state4_T = self.transposeLinear3(hidden_state3_T)    # ***
        batch_hardness_cls_logits = torch.transpose(hidden_state3_T, 1, 2)
        batch_hardness_cls_logits = batch_hardness_cls_logits.squeeze(1)
        # ---------------------------------------------------

        return batch_hardness_cls_logits

    def forward(
            self,
            encoder_input_ids,
            encoder_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table,
            batch_relations=None
    ):
        batch_hardness_cls_logits \
            = self.table_column_cls(
            encoder_input_ids,
            encoder_attention_mask
            # batch_aligned_question_ids,
            # batch_aligned_column_info_ids,
            # batch_aligned_table_name_ids,
            # batch_column_number_in_each_table
        )

        # return {
        #     "batch_table_name_cls_logits": batch_table_name_cls_logits,
        #     "batch_column_info_cls_logits": batch_column_info_cls_logits
        # }
        return {"hardness": batch_hardness_cls_logits}
