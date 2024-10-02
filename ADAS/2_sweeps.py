import wandb
import pandas as pd
import params
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from types import SimpleNamespace
import os, warnings
from utils import get_predictions, create_iou_table, MIOU, BackgroundIOU, \
                  RoadIOU, TrafficLightIOU, TrafficSignIOU, PersonIOU, VehicleIOU, BicycleIOU
from common import label_func
warnings.filterwarnings('ignore')

def get_data(df, bs=4, img_size=(180, 320), augment=True):
    block = DataBlock(blocks=(ImageBlock, MaskBlock(codes=params.BDD_CLASSES)),
                  get_x=ColReader("image_fname"),
                  get_y=ColReader("label_fname"),
                  splitter=ColSplitter(),
                  item_tfms=Resize(img_size),
                  batch_tfms=aug_transforms() if augment else None,
                 )
    return block.dataloaders(df, bs=bs)

def train():
    with wandb.init(entity=params.WANDB_ENTITY, project=params.WANDB_PROJECT, name="train_basic_model", resume=True, job_type="training", config=train_config) as run:
        config = wandb.config
        #################
        # data download #
        #################
        processed_data_at = run.use_artifact(f'{params.WANDB_ENTITY}/{params.WANDB_PROJECT}/{params.PROCESSED_DATA_AT_V1}:latest', type='split_data')
        processed_dataset_dir = Path(processed_data_at.download())
        df = pd.read_csv(processed_dataset_dir / 'data_split.csv')

        #  We do not use a holdout set here. The 'is_valid' column is set
        #  to inform the trainer about the split between training and validation.
        df = df[df.Stage != 'test'].reset_index(drop=True)
        df['is_valid'] = df.Stage == 'valid'
        # assign paths
        # We use the fastai DataBlock API to feed data for the training and validation of the model.
        df["image_fname"] = [processed_dataset_dir/f'images/{f}' for f in df.File_Name.values]
        df["label_fname"] = [label_func(f) for f in df.image_fname.values]

        dls = get_data(df, bs=config.batch_size, img_size=config.img_size, augment=config.augment)

        metrics = [MIOU(), BackgroundIOU(), RoadIOU(), TrafficLightIOU(), \
                TrafficSignIOU(), PersonIOU(), VehicleIOU(), BicycleIOU()]

        learn = unet_learner(dls, arch=resnet18, pretrained=config.pretrained, metrics=metrics)
        ######################
        # set wandb callback #
        ######################
        callbacks = [
            SaveModelCallback(monitor='miou'),
            WandbCallback(log_preds=False, log_model=True)
        ]
        learn.fit_one_cycle(config.epochs, config.lr, cbs=callbacks)

        samples, outputs, predictions = get_predictions(learn)
        table = create_iou_table(samples, outputs, predictions, params.BDD_CLASSES)
        #############
        # log table #
        #############
        wandb.log({"pred_table":table})

        scores = learn.validate()
        metric_names = ['final_loss'] + [f'final_{x.name}' for x in metrics]
        final_results = {metric_names[i] : scores[i] for i in range(len(scores))}
        for k,v in final_results.items():
            wandb.summary[k] = v
        # We are reloading the model from the best checkpoint at the end and saving it.
        # To make sure we track the final metrics correctly,
        # we will validate the model again and save the final loss and metrics to wandb.summary.

if __name__ == "__main__":
    import params
    wandb.login()
    train_config = SimpleNamespace(
        framework="fastai",
        img_size=(180, 320),
        batch_size=4,
        augment=True, # use data augmentation
        epochs=20,
        lr=2e-2,
        pretrained=True,  # whether to use pretrained encoder
        seed=42,
    )

    set_seed(train_config.seed, reproducible=True)
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'miou',
            'goal': 'maximize'
        },
        'parameters': {
            'optimizer': {
                'values': ['adamw_torch', 'sgd']
            },
            'lr': {
                'distribution': 'uniform',
                'min': 2e-3,
                'max': 2e-2
            },
            'batch_size': {
                'values': [16, 32]
            },
            'epoch': {
                'value': 20
            },
            'img_size': {
                'value': (180, 320)
            },
            'augment': {
                'value': True
            },
            'pretrained': {
                'value': True
            },
            'seed': {
                'value': 42
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=params.WANDB_PROJECT)
    wandb.agent(sweep_id, train, count=5)