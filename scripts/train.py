import torch
from tqdm import tqdm
from loss import bce
from metrics import precision, recall
import wandb
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_script(model, train_set, val_set, epochs, multiple_gpus, save_path):
    wandb.init(project = 'mayo_data')
    dataloaders = {
        'train' : train_set,
        'val' : val_set
    }
    if not next(model.parameters()).is_cuda:
        model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        train_loss, train_prec, train_rec = 0.0, 0.0, 0.0
        val_loss, val_prec, val_rec = 0.0, 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_prec, running_rec = 0.0, 0.0, 0.0
            with tqdm(dataloaders[phase], unit='batch') as tepoch:
                for img, label in dataloaders[phase]:
                    img = img.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(img)
                        loss = bce(outputs, label)
                        prec = precision(outputs, label)
                        rec = recall(outputs, label)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_prec += prec.item()
                    running_rec += rec.item()
                    tepoch.set_postfix(loss = loss.item(), prec = prec.item(), rec = rec.item())
            if phase == 'train':
                train_loss = running_loss / len(train_set)
                train_prec = running_prec / len(train_set)
                train_rec = running_rec / len(train_set)
                print(f'Train loss: {train_loss}')
                print(f'Train prec: {train_prec}')
                print(f'Train rec: {train_rec}')
            else:
                val_loss = running_loss / len(val_set)
                val_prec = running_prec / len(val_set)
                val_rec = running_rec / len(val_set)
                print(f'Train loss: {val_loss}')
                print(f'Train prec: {val_prec}')
                print(f'Train rec: {val_rec}')
        wandb.log({
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'train_prec' : train_prec,
            'val_prec' : val_prec,
            'train_rec' : train_rec,
            'val_rec' : val_rec
        })
        save_model(model, epoch, optimizer, multiple_gpus, save_path)


def save_model(model, epoch, optimizer, multiple_gpus, save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    # Please refer this link for saving and loading models
    # Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save({
        'epoch' : epoch,
        # Please change this..... if you are using single GPU to model.state_dict().
        'model_state_dict': model.module.state_dict() if multiple_gpus == True else model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }, f'{save_path}{epoch}.pth')
    print(f'Weight saved for epoch {epoch}.')