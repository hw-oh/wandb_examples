WANDB_ENTITY = "wandb-korea"#"jyjang-miridih"
WANDB_PROJECT = "ADAS-handson"
BDD_CLASSES = {i:c for i,c in enumerate(['background', 'road', 'traffic light', 'traffic sign', 'person', 'vehicle', 'bicycle'])}
RAW_DATA_AT = 'bdd_simple_1k'
PROCESSED_DATA_AT_V1 = 'bdd_simple_1k_split_v1'
PROCESSED_DATA_AT_V2 = 'bdd_simple_1k_split_v2'
PROCESSED_DATA_AT_V3 = 'bdd_simple_1k_split_v3'