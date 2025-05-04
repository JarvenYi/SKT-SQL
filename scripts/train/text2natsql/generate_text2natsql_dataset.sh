set -e

# generate text2natsql training dataset with noise_rate 0.2
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/1_preprocessed_train_structure_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/2_sktsql_train_structure_hds_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "train" \
    --noise_rate 0.2 \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# predict probability for each schema item in the eval set
python schema_item_classifier.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "./models/schema_items_natsql" \
    --dev_filepath "./data/preprocessed_data/1_preprocessed_dev_structure_natsql.json" \
    --output_filepath "./data/preprocessed_data/2i_schema_decoupler_dev_structure_hds_natsql.json" \
    --use_contents \
    --mode "eval"

# generate text2natsql development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/2i_schema_decoupler_dev_structure_hds_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/2_refined_schema_dev_structure_hds_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"