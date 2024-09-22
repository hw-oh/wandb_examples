import torch, pickle
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb
from params import ENTITY, PROJECT, DATA, CONFIG

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return model, total_loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

def main(CONFIG, training_data, test_data):
    # Training and test
    with wandb.init(entity=ENTITY, project=PROJECT) as run:
        wandb.config.update(CONFIG)
        # Create data loaders.
        train_dataloader = DataLoader(training_data, batch_size=CONFIG['batch_size'])
        test_dataloader = DataLoader(test_data, batch_size=CONFIG['batch_size'])

        for X, y in test_dataloader:
            print(f"Shape of X [N, C, H, W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

        # Get cpu or gpu device for training.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = NeuralNetwork().to(device)
        print(f"Using {device} device")
        print(model)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
        epochs = CONFIG['epochs']
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            model, total_loss = train(train_dataloader, model, loss_fn, optimizer, device)
            correct, test_loss = test(test_dataloader, model, loss_fn, device)
            #####################################################################
            wandb.log({"train_loss": total_loss / len(train_dataloader)}, step=t)
            wandb.log({"test_loss": test_loss, "test_acc": correct}, step=t)
            #####################################################################

        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")

        classes = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

        model.eval()
        x, y = test_data[0][0], test_data[0][1]
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]

        #####################################################################
        table = wandb.Table(columns=["Image", "Predicted", "Actual"])
        for i in range(10):
            x, y = test_data[i][0], test_data[i][1]
            with torch.no_grad():
                pred = model(x)
                predicted, actual = classes[pred[0].argmax(0)], classes[y]
                table.add_data(wandb.Image(x), predicted, actual)
        wandb.log({"results": table})

        model_artifact = wandb.Artifact("SimpleNN", type="model")
        model_artifact.add_file("model.pth")
        run.log_artifact(model_artifact)
        #####################################################################

if __name__ == "__main__":

    # Load the artifact
    with wandb.init() as run:
        artifact = run.use_artifact(f'{ENTITY}/{PROJECT}/{DATA}:latest', type='dataset')
        artifact_dir = artifact.download()

    with open(f"{artifact_dir}/fmnist_training.pkl", "rb") as f:
        training_data = pickle.load(f)
    with open(f"{artifact_dir}/fmnist_test.pkl", "rb") as f:
        test_data = pickle.load(f)

    main(CONFIG, training_data, test_data)