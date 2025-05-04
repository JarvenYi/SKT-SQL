from .process_dataset import preprocessing_generate_lgerels
from .lgerels2t5rels import preprocessing_lgerels2t5rels
from .lgerels2t5rels_changeOrder import preprocessing_lgerels2t5rels_changeOrder
from .get_relation2id_dict import get_relation2id_dict


def preprocess_by_dataset(data_base_dir, dataset_name, t5_processed, mode, edge_type="Default", use_coref=False, use_dependency = False):
    # edgeType: Default, MinType
    # edgeType = "Default"
    # use_coref = True
    # use_dependency =True
    '''
    preprocessing_generate_lgerels():
    用来生成question-relation, schema linking(table-column relation & schema-question relation)
    并将其存入./dataset_files/preprocessed_dataset/spider/tables.bin文件中
    值得一提的是，此处的use_dependency并没有任何动作.        - Jarven
    '''
    preprocessing_generate_lgerels(data_base_dir, dataset_name, mode, use_coref, use_dependency)

    # _, relations=preprocessing_lgerels2t5rels(data_base_dir, dataset_name, t5_processed, mode, edgeType, use_coref)
    _, relations = preprocessing_lgerels2t5rels_changeOrder(data_base_dir, dataset_name, t5_processed, mode, edge_type, use_coref, use_dependency)
    return relations