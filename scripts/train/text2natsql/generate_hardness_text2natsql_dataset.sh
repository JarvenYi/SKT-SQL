set -e


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