# SKT-SQL: Multi-Decoupling and Skeleton Prompt Framework for Text-to-SQL on Small-Scale LMs 
### Environment Setup, Spider 1.0 Dataset, and Model Download
Environment: Python 3.8, Torch 1.11.0, CUDA 12.4, and related dependencies specified in requirements.txt.

Please Download Spider 1.0 data and database and then unzip them.
Put Spider 1.0 "train_spider.json", "dev.json" and "tables.json" into "./data/spider/" 
Put Spider 1.0 database folder into "./database"

## 1. Training SKT-SQL
### Preprocessing
Execute the following terminal command in the corresponding location: \
`sh ./scripts/train/text2natsql/preprocess.sh`\
Output:\
Training data: "./data/preprocessed_data/1_preprocessed_train_structure_natsql.json.json" \
Validation data (input $X'$): "./data/preprocessed_data/1_preprocessed_dev_structure_natsql.json" \
### Training Schema Decoupler
Execute the terminal command: \
`sh ./scripts/train/text2natsql/train_text2sql_schema_item_classifier.sh` \
This will generate the schema ranker model at "./models/text2sql_schema_item_sql".\

### Obtaining Refined Schema via Schema Decoupler
Execute the terminal command: \
`sh ./scripts/train/text2natsql/generate_text2sql_dataset.sh` \
Output: \
Refined schema for training: "./data/preprocessed_data/2_sktsql_train_structure_hds_natsql" \
Refined schema for validation (input $X_s$): "./data/preprocessed_data/2_refined_schema_dev_structure_hds_natsql.json" \
### Training Hardness Decoupler
Execute the terminal command: \
`sh ./scripts/train/text2sql/train_text2natsql_hardness_classifier.sh` \
This will generate the hardness classifier model at "./models/text2sql_hardness_classifier_Jarven/RoBerta-large/4Linear".\

### Obtaining Hardness Labels via Hardness Decoupler
Execute the terminal command: \
`sh ./scripts/train/text2natsql/generate_hardness_data_generator.sh` \
(input $X_h$) save at : "./data/preprocessed_data/3_hds_classified_schema_dev_structure_hds_natsql.json"\

### Training and Evaluting Skeleton Generator
Execute the following terminal commands separately to train and test the four distributed generators: 
`sh scripts/train/text2natsql/train_text2natskt.sh` \
save skeleton_model at "./models/Skeletons-structure-hds-large-natsql" \
the result of evalution (input $X_k$) save at "./data/preprocessed_data/4_predicted_Skeletons-structure-hds-large-natsql"  \

#### Training and Evaluting SQL Generator:
`sh scripts/train/text2natsql/train_text2natsql_t5_large.sh` \
Output: Overall EM and EX metrics for SKT-SQL and for each hardness sample.\
save sql_model at "./models/SQL_Generator-large-structure-hds-natsql"\
NOTE: select model path to finish evaluation\

__NOTE: If you encounter an error IndexError: list index out of range, please delete the record in the corresponding folder in `./eval_results/.`__

# 2. Validating SKT-SQL
Execute the terminal command: \
`sh ./scripts/inference/text2sql.sh scale_level spider` \
or \
`sh ./scripts/inference/text2natsql.sh scale_level spider` \
where `scale_level` can be selected from `base`, `large`.