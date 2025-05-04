set -e

device="0"
tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

Skeletons_model_save_path="./models/Skeletons-structure-hds-large-natsql-199920/checkpoint-199920"

if [ $1 = "base" ]
then
    SKT_SQL_model_save_path="./models/SQL_Generator-base-structure-hds-natsql-XXX"
    text2natsql_model_bs=12
elif [ $1 = "large" ]
then
    SKT_SQL_model_save_path="./models/SQL_Generator-large-structure-hds-natsql-XXX"
    text2natsql_model_bs=6
elif [ $1 = "3b" ]
then
    SKT_SQL_model_save_path="./models/SQL_Generator-3b-structure-hds-natsql-XXX"
    text2natsql_model_bs=1
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

model_name="sktsql_$1"

if [ $2 = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider/dev.json"
    db_path="./database"
    output="./predictions/Spider-dev/$model_name/pred.sql"
else
    echo "The second arg must in [spider]."
    exit
fi

# prepare table file for natsql
# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path $table_path \
    --input_dataset_path $input_dataset_path \
    --natsql_dataset_path "./NatSQL/NatSQLv1_6/dev-natsql.json" \
    --output_dataset_path "./data/preprocessed_data/1_preprocessed_dev_structure_natsql.json" \
    --db_path $db_path \
    --target_type "natsql" \
    --abstract_level 2

# preprocess tables.json for natsql
python NatSQL/table_transform.py \
    --in_file $table_path \
    --out_file $tables_for_natsql \
    --correct_col_type \
    --remove_start_table  \
    --analyse_same_column \
    --table_transform \
    --correct_primary_keys \
    --use_extra_col_types \
    --db_path $db_path

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

# predict probability for each schema item in the eval set
python hardness_classifier.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2natsql_hardness_classifier_Jarven/RoBerta-large/4Linear-resd" \
    --dev_filepath "./data/preprocessed_data/2_refined_schema_dev_structure_hds_natsql.json" \
    --output_filepath "./data/preprocessed_data/3_hds_classified_schema_dev_structure_hds_natsql.json" \
    --use_contents \
    --resd_preprocessed \
    --mode "eval"

# DEV skeleton generation
python text2skt.py \
    --batch_size 8 \
    --gradient_descent_step 2 \
    --device "1" \
    --learning_rate 1e-4 \
    --epochs 1 \
    --seed 42 \
    --save_path $Skeletons_model_save_path \
    --tensorboard_save_path "./tensorboard_log/Skeletons-structure-hds-large-natsql" \
    --model_name_or_path "skt-structure-hds-base-natsql" \
    --use_adafactor \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/3_hds_classified_schema_dev_structure_hds_natsql.json" \
    --original_dev_filepath "data/spider/dev.json" \
    --db_path "database" \
    --tables_for_natsql "NatSQL/NatSQLv1_6/tables_for_natsql.json" \
    --target_type "natsql" \
    --output "./data/preprocessed_data/4_predicted_Skeletons-structure-hds-large-natsql"


# select the best text2natsql-t5-large ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size $text2natsql_model_bs \
    --device $device \
    --seed 42 \
    --save_path $SKT_SQL_model_save_path \
    --eval_results_path "./eval_results/SQL_Generator-large-structure-hds-natsql" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/4_predicted_Skeletons-structure-hds-large-natsql-199920.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output "./predictions/natsql-SQL_query-SQL_Generator-large-structure-hds+Skeletons-structure-hds-large-199920.txt"

#Evaluation
python -u evaluate_total.py \
    --easy_model_sql "./predictions/natsql-SQL_query-SQL_Generator-large-structure-hds+Skeletons-structure-hds-large-199920.txt" \
    --easy_original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database"