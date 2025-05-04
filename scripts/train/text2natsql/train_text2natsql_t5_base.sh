set -e

# train text2natsql-t5-base model
python -u text2sql.py \
    --batch_size 12 \
    --gradient_descent_step 2 \
    --device "0" \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/SQL_Generator-base-structure-hds-natsql" \
    --tensorboard_save_path "./tensorboard_log/SQL_Generator-base-structure-hds-natsql" \
    --model_name_or_path "t5-base" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/2_sktsql_train_structure_hds_natsql.json"

# select the best text2natsql-t5-large ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 16 \
    --device "1" \
    --seed 42 \
    --save_path "./models/SQL_Generator-base-structure-hds-natsql" \
    --eval_results_path "./eval_results/SQL_Generator-base-structure-hds-natsql" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/4_predicted_Skeletons-structure-hds-large-natsql-xxx.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql" \
    --output "./predictions/natsql-SQL_query-SQL_Generator-base-structure-hds+Skeletons-structure-hds-large-xxx.txt"

#Evaluation
python -u evaluate_total.py \
    --easy_model_sql './predictions/natsql-SQL_query-SQL_Generator-base-structure-hds+Skeletons-structure-hds-large-xxx.txt' \
    --easy_original_dev_filepath './data/spider/dev.json' \
    --db_path "./database"