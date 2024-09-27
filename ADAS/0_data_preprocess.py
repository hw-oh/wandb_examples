import wandb
import os



import pandas as pd
import params
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from types import SimpleNamespace
import os, warnings
from sklearn.model_selection import StratifiedGroupKFold
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU
from common import label_func, get_classes_per_image, _create_table

def download_raw_data(path):
    # Download raw data
    with wandb.init(entity=params.WANDB_ENTITY, project=params.WANDB_PROJECT, job_type="upload", name="create-raw-data-artifact") as run:

        # log data with Artifacts
        raw_data_at = wandb.Artifact(params.RAW_DATA_AT,
                                    type="raw_data",
                                    metadata={
                                        "url": 'https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip',
                                    })
        raw_data_at.add_file(path/'LICENSE.txt', name='LICENSE.txt')
        raw_data_at.add_dir(path/'images', name='images')
        raw_data_at.add_dir(path/'labels', name='labels')

        # Visualize data with Tables
        image_files = get_image_files(path/"images", recurse=False)
        if DEBUG: image_files = image_files[:10]
        table = _create_table(image_files, params.BDD_CLASSES)
        raw_data_at.add(table, "eda_table")
        run.log_artifact(raw_data_at)

def preprocess(path):
    for PROCESSED_DATA_AT in [params.PROCESSED_DATA_AT_V1, params.PROCESSED_DATA_AT_V2, params.PROCESSED_DATA_AT_V3]:
        with wandb.init(entity=params.WANDB_ENTITY, project=params.WANDB_PROJECT, job_type="data_split", name=f"create-{PROCESSED_DATA_AT}") as run:
            ## Data Preparation
            # data download
            raw_data_at = run.use_artifact(f'{params.WANDB_ENTITY}/{params.WANDB_PROJECT}/{params.RAW_DATA_AT}:latest')
            path = Path(raw_data_at.download())

            fnames = os.listdir(path/'images')
            groups = [s.split('-')[0] for s in fnames]
            orig_eda_table = raw_data_at.get("eda_table")
            y = orig_eda_table.get_column('bicycle')

            df = pd.DataFrame()
            df['File_Name'] = fnames
            df['fold'] = -1

            # data split
            cv = StratifiedGroupKFold(n_splits=10)
            for i, (train_idxs, test_idxs) in enumerate(cv.split(fnames, y, groups)):
                df.loc[test_idxs, ['fold']] = i

            df['Stage'] = 'train'
            df.loc[df.fold == 0, ['Stage']] = 'test'
            df.loc[df.fold == 1, ['Stage']] = 'valid'
            del df['fold']
            df.Stage.value_counts()
            df.to_csv('data_split.csv', index=False)
            processed_data_at = wandb.Artifact(PROCESSED_DATA_AT, type="split_data")

            # Data upload
            processed_data_at.add_file('data_split.csv')
            processed_data_at.add_dir(path)

            # Data Visualization
            # Table for eda
            data_split_table = wandb.Table(dataframe=df[['File_Name', 'Stage']])

            # join table
            join_table = wandb.JoinedTable(orig_eda_table, data_split_table, "File_Name")
            processed_data_at.add(join_table, "eda_table_data_split")
            run.log_artifact(processed_data_at) # visualization on wandb artifacts

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    wandb.login()

    DEBUG = False
    URL = 'https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip'
    path = Path(untar_data(URL, force_download=True))
    download_raw_data(path)
    preprocess(path)
