set -e

# training skt model based on t5
python text2skt.py \
 --batch_size 4 \
 --gradient_descent_step 2 \
 --device "1" \
 --learning_rate 1e-4 \
 --epochs 128 \
 --seed 42 \
 --save_path "./models/Skeletons-structure-hds-large-natsql" \
 --tensorboard_save_path "./tensorboard_log/Skeletons-structure-hds-large-natsql" \
 --model_name_or_path "t5-large" \
 --use_adafactor \
 --mode train \
 --train_filepath "./data/preprocessed_data/2_sktsql_train_structure_hds_natsql.json" \
 --hardness_prompt

# evaluating trained skt model
python evaluate_text2skt_ckpts.py \
  --batch_size 6  \
  --device "1"\
  --seed 42 \
  --save_path "./models/Skeletons-structure-hds-large-natsql" \
  --eval_results_path "./eval_results/Skeletons-structure-hds-large-natsql" \
  --mode eval \
  --dev_filepath "./data/preprocessed_data/3_hds_classified_schema_dev_structure_hds_natsql.json"\
  --original_dev_filepath "./data/spider/dev.json"\
  --db_path "./database"\
  --num_beams 8\
  --num_return_sequences 8\
  --target_type "natsql" \

# --------------------DEV (select a skeleton-model path into "save_path")-----------------------
build dev data by skt model based on t5
python text2skt.py \
   --batch_size 8 \
   --gradient_descent_step 2 \
   --device "1" \
   --learning_rate 1e-4 \
   --epochs 1 \
   --seed 42 \
   --save_path "./models/Skeletons-structure-hds-large-natsql/checkpoint-xxx" \
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
