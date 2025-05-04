import json
from torch.utils.data import Dataset

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")
HARDNESS = {
    "component1": ("where", "group", "order", "limit", "join", "or", "like"),
    "component2": ("except", "union", "intersect"),
}
LEVELS = ["easy", "medium", "hard", "extra", "all"]


def has_agg(unit):
    return unit[0] != AGG_OPS.index("none")


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def count_component1(sql):
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len(
        [
            cond_unit
            for cond_unit in cond_units
            if cond_unit[1] == WHERE_OPS.index("like")
        ]
    )

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql["select"][1])
    agg_count += count_agg(sql["where"][::2])
    agg_count += count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql["select"][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql["where"]) > 1:
        count += 1

    # number of group by clauses
    if len(sql["groupBy"]) > 1:
        count += 1

    return count


def eval_hardness(sql):
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "easy", 0
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium", 1
    elif (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard", 2
    else:
        return "extra", 3


class ColumnAndTableClassifierDataset(Dataset):
    def __init__(
            self,
            dir_: str = None,
            use_contents: bool = True,
            add_fk_info: bool = True,
            relations=None
    ):
        super(ColumnAndTableClassifierDataset, self).__init__()

        self.questions: list[str] = []

        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []
        # ------------- Relation matrix and DB id -----------------
        if relations is not None:
            self.relation_matrix = relations
        else:
            self.relation_matrix = None
        # ---------------------------------------
        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]

                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)

            if add_fk_info:
                table_column_id_list = []
                # add a [FK] identifier to foreign keys
                for fk in data["fk"]:
                    source_table_name_original = fk["source_table_name_original"]
                    source_column_name_original = fk["source_column_name_original"]
                    target_table_name_original = fk["target_table_name_original"]
                    target_column_name_original = fk["target_column_name_original"]

                    if source_table_name_original in table_names_original_in_one_db:
                        source_table_id = table_names_original_in_one_db.index(source_table_name_original)
                        source_column_id = column_names_original_in_one_db[source_table_id].index(
                            source_column_name_original)
                        if [source_table_id, source_column_id] not in table_column_id_list:
                            table_column_id_list.append([source_table_id, source_column_id])

                    if target_table_name_original in table_names_original_in_one_db:
                        target_table_id = table_names_original_in_one_db.index(target_table_name_original)
                        target_column_id = column_names_original_in_one_db[target_table_id].index(
                            target_column_name_original)
                        if [target_table_id, target_column_id] not in table_column_id_list:
                            table_column_id_list.append([target_table_id, target_column_id])

                for table_id, column_id in table_column_id_list:
                    if extra_column_info_in_one_db[table_id][column_id] != "":
                        extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
                    else:
                        extra_column_info_in_one_db[table_id][column_id] += "[FK]"

            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos_in_one_table = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id],
                                                          extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos_in_one_table.append(column_name)
                column_infos_in_one_db.append(column_infos_in_one_table)

            self.questions.append(data["question"])

            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]
        # -------------
        if self.relation_matrix is not None:
            relation_matrix = self.relation_matrix[index]
            return question, table_names_in_one_db, table_labels_in_one_db, \
                   column_infos_in_one_db, column_labels_in_one_db, \
                   relation_matrix
        else:
            return question, table_names_in_one_db, table_labels_in_one_db, \
                   column_infos_in_one_db, column_labels_in_one_db
        # -------------


# --------------------Hardness-----------------------------
class HardnessClassifierDataset(Dataset):
    def __init__(
            self,
            dir_: str = None,
            use_contents: bool = True,
            add_fk_info: bool = True,
            relations=None
    ):
        super(HardnessClassifierDataset, self).__init__()

        self.questions: list[str] = []

        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []

        self.all_hardness_names: list[str] = []
        self.all_hardness_labels: list[int] = []

        # ------------- Relation matrix and DB id -----------------
        if relations is not None:
            self.relation_matrix = relations
        else:
            self.relation_matrix = None
        # ---------------------------------------
        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]

                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)

            if add_fk_info:
                table_column_id_list = []
                # add a [FK] identifier to foreign keys
                for fk in data["fk"]:
                    source_table_name_original = fk["source_table_name_original"]
                    source_column_name_original = fk["source_column_name_original"]
                    target_table_name_original = fk["target_table_name_original"]
                    target_column_name_original = fk["target_column_name_original"]

                    if source_table_name_original in table_names_original_in_one_db:
                        source_table_id = table_names_original_in_one_db.index(source_table_name_original)
                        source_column_id = column_names_original_in_one_db[source_table_id].index(
                            source_column_name_original)
                        if [source_table_id, source_column_id] not in table_column_id_list:
                            table_column_id_list.append([source_table_id, source_column_id])

                    if target_table_name_original in table_names_original_in_one_db:
                        target_table_id = table_names_original_in_one_db.index(target_table_name_original)
                        target_column_id = column_names_original_in_one_db[target_table_id].index(
                            target_column_name_original)
                        if [target_table_id, target_column_id] not in table_column_id_list:
                            table_column_id_list.append([target_table_id, target_column_id])

                for table_id, column_id in table_column_id_list:
                    if extra_column_info_in_one_db[table_id][column_id] != "":
                        extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
                    else:
                        extra_column_info_in_one_db[table_id][column_id] += "[FK]"

            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos_in_one_table = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id],
                                                          extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos_in_one_table.append(column_name)
                column_infos_in_one_db.append(column_infos_in_one_table)
            self.questions.append(data["question"])
            # if resd is False:
            #     self.questions.append(data["question"])
            # elif resd is True:
            #     self.questions.append(data["input_sequence"])

            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)

            self.all_hardness_names.append(data["hardness"])
            self.all_hardness_labels.append(data["hardness_labels"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]

        hardness_names_in_one_db = self.all_hardness_names[index]
        hardness_labels_in_one_db = self.all_hardness_labels[index]
        # -------------
        if self.relation_matrix is not None:
            relation_matrix = self.relation_matrix[index]
            return question, table_names_in_one_db, table_labels_in_one_db, \
                   column_infos_in_one_db, column_labels_in_one_db, \
                   hardness_names_in_one_db, hardness_labels_in_one_db, \
                   relation_matrix
        else:
            return question, table_names_in_one_db, table_labels_in_one_db, \
                   column_infos_in_one_db, column_labels_in_one_db, \
                   hardness_names_in_one_db, hardness_labels_in_one_db
        # -------------


class HardnessClassifierDatasetForRESD(Dataset):
    def __init__(
            self,
            dir_: str = None,
            relations=None
    ):
        super(HardnessClassifierDatasetForRESD, self).__init__()

        self.input_sequence: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        self.all_hardness_names: list[str] = []
        self.all_hardness_labels: list[int] = []

        # ------------- Relation matrix and DB id -----------------
        if relations is not None:
            self.relation_matrix = relations
        else:
            self.relation_matrix = None
        # ---------------------------------------
        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            self.input_sequence.append(data["input_sequence"])
            self.all_hardness_labels.append(data["hardness_labels"])

            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

    def __len__(self):
        return len(self.input_sequence)

    def __getitem__(self, index):
        input_sequence = self.input_sequence[index]
        hardness_labels_in_one_db = self.all_hardness_labels[index]
        # -------------
        if self.relation_matrix is not None:
            relation_matrix = self.relation_matrix[index]
            return input_sequence, hardness_labels_in_one_db, \
                   relation_matrix

        else:
            return input_sequence, hardness_labels_in_one_db
        # -------------


# -----------------------Jarven---------------------
class ColumnAndTableRelationDataset(Dataset):
    def __init__(
            self,
            dir_: str = None,
            use_contents: bool = True,
            add_fk_info: bool = True
    ):
        super(ColumnAndTableRelationDataset, self).__init__()

        self.questions: list[str] = []

        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []

        self.db_id: list[str] = []
        # self.relation : list[str] = []
        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            # relation = []
            db_id = []
            db_id.append(data["db_id"])  # capture db_id
            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]

                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)

            # if add_fk_info:
            #     table_column_id_list = []
            #     # add a [FK] identifier to foreign keys
            #     for fk in data["fk"]:
            #         source_table_name_original = fk["source_table_name_original"]
            #         source_column_name_original = fk["source_column_name_original"]
            #         target_table_name_original = fk["target_table_name_original"]
            #         target_column_name_original = fk["target_column_name_original"]
            #
            #         if source_table_name_original in table_names_original_in_one_db:
            #             source_table_id = table_names_original_in_one_db.index(source_table_name_original)
            #             source_column_id = column_names_original_in_one_db[source_table_id].index(
            #                 source_column_name_original)
            #             if [source_table_id, source_column_id] not in table_column_id_list:
            #                 table_column_id_list.append([source_table_id, source_column_id])
            #
            #         if target_table_name_original in table_names_original_in_one_db:
            #             target_table_id = table_names_original_in_one_db.index(target_table_name_original)
            #             target_column_id = column_names_original_in_one_db[target_table_id].index(
            #                 target_column_name_original)
            #             if [target_table_id, target_column_id] not in table_column_id_list:
            #                 table_column_id_list.append([target_table_id, target_column_id])
            #
            #     for table_id, column_id in table_column_id_list:
            #         if extra_column_info_in_one_db[table_id][column_id] != "":
            #             extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
            #         else:
            #             extra_column_info_in_one_db[table_id][column_id] += "[FK]"

            # column_info = column name + extra column info
            column_infos_in_one_db = []
            # for table_id in range(len(table_names_in_one_db)):
            #     column_infos_in_one_table = []
            #     for column_name, extra_column_info in zip(column_names_in_one_db[table_id],
            #                                               extra_column_info_in_one_db[table_id]):
            #         if len(extra_column_info) != 0:
            #             column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
            #         else:
            #             column_infos_in_one_table.append(column_name)
            #     column_infos_in_one_db.append(column_infos_in_one_table)

            self.questions.append(data["question"])

            self.all_table_names.append(table_names_original_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_names_original_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)
            self.db_id.append(db_id)
            # self.relation.append(relation)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]

        db_id = self.db_id[index]
        # relation = self.relation[index]

        return question, table_names_in_one_db, table_labels_in_one_db, \
               column_infos_in_one_db, column_labels_in_one_db, db_id


class Text2SQLDataset(Dataset):
    def __init__(
            self,
            dir_: str,
            mode: str
    ):
        super(Text2SQLDataset).__init__()

        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode == "train":
                self.output_sequences.append(data["output_sequence"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index], self.all_tc_original[
                index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]


# ---------------------- Skeleton -------------------------
class Text2SkeletonDataset(Dataset):
    def __init__(
            self,
            dir_: str,
            mode: str
    ):
        super(Text2SQLDataset).__init__()

        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_skeleton: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            # remove skeletons tokens in the input_sequence
            if self.mode == 'train':
                data_input_wo_skt = data["input_sequence"].split('<END>')[0] + '<END> '
                self.input_sequences.append(data["input_sequence"].replace(data_input_wo_skt, ''))
            else:
                self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode == "train":
                self.output_skeleton.append(data["skeleton"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_skeleton[index], self.db_ids[index], self.all_tc_original[
                index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]


class Text2SkeletonWithHdsDataset(Dataset):
    def __init__(
            self,
            dir_: str,
            mode: str
    ):
        super(Text2SQLDataset).__init__()

        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_skeleton: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            # remove skeletons tokens in the input_sequence
            if self.mode == 'train':
                data_input_wo_skt = data["input_sequence"].split('<END>')[0] + '<END> '  # ****5-25

                hardness_labels = data["hardness_labels"]
                if hardness_labels == 0:
                    self.input_sequences.append("[/easy] " + data["input_sequence"].replace(data_input_wo_skt, ''))
                elif hardness_labels == 1:
                    self.input_sequences.append("[/medium] " + data["input_sequence"].replace(data_input_wo_skt, ''))
                elif hardness_labels == 2:
                    self.input_sequences.append("[/hard] " + data["input_sequence"].replace(data_input_wo_skt, ''))
                elif hardness_labels == 3:
                    self.input_sequences.append("[/extra] " + data["input_sequence"].replace(data_input_wo_skt, ''))
            else:
                self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode == "train":
                self.output_skeleton.append(data["skeleton"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_skeleton[index], self.db_ids[index], self.all_tc_original[
                index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]


class Text2SQLWithHdsDataset(Dataset):
    def __init__(
            self,
            dir_: str,
            mode: str
    ):
        super(Text2SQLDataset).__init__()

        self.mode = mode

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        for data in dataset:
            if self.mode == 'train':
                data_input_skt = data["input_sequence"].split('<END>')[0] + '<END>'
                # data_input_skt.replace("<END> ", "")
                data_input_wo_skt = data["input_sequence"].split('<END>')[1]
                data_input_wo_skt.replace("<END> ", "")
                hardness_labels = data["hardness_labels"]
                if hardness_labels == 0:
                    self.input_sequences.append(data_input_skt + " [/easy]" + data_input_wo_skt)
                elif hardness_labels == 1:
                    self.input_sequences.append(data_input_skt + " [/medium]" + data_input_wo_skt)
                elif hardness_labels == 2:
                    self.input_sequences.append(data_input_skt + " [/hard]" + data_input_wo_skt)
                elif hardness_labels == 3:
                    self.input_sequences.append(data_input_skt + " [/extra]" + data_input_wo_skt)
            else:
                self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            if self.mode == "train":
                self.output_sequences.append(data["output_sequence"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index], self.all_tc_original[
                index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]