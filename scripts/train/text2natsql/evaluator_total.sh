set -e

python -u evaluate_total.py \
    --easy_model_sql 'Easy_DQCP_3B_predicted_natsql.txt' \
    --medium_model_sql 'Medium_DQCP_3B_predicted_natsql.txt' \
    --hard_model_sql 'Hard_DQCP_3B_predicted_natsql.txt' \
    --extra_hard_model_sql 'Extra-hard_DQCP_3B_predicted_natsql.txt' \
    --easy_original_dev_filepath './data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Easy_spider_dev_dataset.json' \
    --medium_original_dev_filepath './data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Medium_spider_dev_dataset.json' \
    --hard_original_dev_filepath './data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Hard_spider_dev_dataset.json' \
    --extra_original_dev_filepath './data/preprocessed_data/preprocessed_by_resd_hardness_with_prob_natsql/Extra-hard_spider_dev_dataset.json' \
    --db_path './database'