set -e

# train schema item classifier
python -u hardness_classifier.py \
    --batch_size 8 \
    --gradient_descent_step 2 \
    --device "1" \
    --learning_rate 1e-5 \
    --gamma 2.0 \
    --alpha 0.75 \
    --epochs 64 \
    --patience 16 \
    --seed 42 \
    --save_path "./models/text2natsql_hardness_classifier_Jarven/RoBerta-large/4Linear-resd" \
    --tensorboard_save_path "./tensorboard_log/text2natsql_hardenss_classifier_Jarven/hardness-resd_RoBERTa-large-4Linear" \
    --train_filepath "./data/preprocessed_data/2_sktsql_train_structure_hds_natsql.json" \
    --dev_filepath "./data/preprocessed_data/2_refined_schema_dev_structure_hds_natsql.json" \
    --model_name_or_path "roberta-large" \
    --resd_preprocessed_data \
    --use_contents \
    --mode "train"\
    --edge_type "Default" \
    --output_filepath "./data/preprocessed_data/hardness_dev_with_probs_natsql.json" \
