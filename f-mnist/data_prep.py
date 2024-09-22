import pickle
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb
import os
from params import ENTITY, PROJECT, DATA

# Preprocess data
## Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

## Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
# Create directory for prep_data if it doesn't exist
os.makedirs("./prep_data", exist_ok=True)
with open("./prep_data/fmnist_training.pkl", "wb") as f:
    pickle.dump(training_data, f)
with open("./prep_data/fmnist_test.pkl", "wb") as f:
    pickle.dump(test_data, f)
    
#####################################################################
## Make an artifact for the data
with wandb.init(entity=ENTITY, project=PROJECT, name='prepare_data') as run:
    mnist = wandb.Artifact(DATA, type="dataset")
    mnist.add_dir("prep_data")
    run.log_artifact(mnist)
#####################################################################