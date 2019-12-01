"""
autor: my new life 
copyleft
classificateur ecrit pous une classification d'image en noir et blanc
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

##definition du model a entrainer 
class Classifieur(nn.Module):
    def __init__(self, nbentre, nbclass):
        super().__init__()
        #on a un model en 4 couches, une couche d'entree a nbentre points, deux intermediaires, une couche de sortie de taille nbclass
        #la couche de sortie
        self.fc1 = nn.Linear(nbentre, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, nbclass)

    def forward(self, x):
        x  = x.view(x.shape[0], -1)
        x  = F.relu(self.fc1(x))
        x  = F.relu(self.fc2(x))
        x  = F.relu(self.fc3(x))
        x  = F.log_softmax(self.fc4(x), dim=1)
        return x

def load_data(dirname):
    ##on applique aux images des transformations 
    #transformation pour le jeux d'entrainement 
    train_transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224), transforms.ToTensor()])

    #transformation pour le jeux de test 
    test_transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
    
    #les donnees sont enregistree 
    train_data = datasets.ImageFolder(dirname+'/train', transform=train_transform)
    test_data  = datasets.ImageFolder(dirname+'/train', transform=test_transform)

    #les loaders qui permetent
    loader_data_train = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    loader_data_test = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    return loader_data_train, loader_data_test

def entrainement(dirname, dim, nbclass, nbtour = 5):
    """
    dirname est le nom du dossier contenant les photos 
    dim est une liste a deux elements
    nbclass nombre de classes classifier
    """
    loader_train, _ = load_data(dirname)
    #le classifeur attend les dimmensions de la photo et le dataset
    model = Classifieur(dim[1]*dim[0], nbclass)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    #nombre de round d'entrainement
    epochs = 5

    for _ in range(epochs):
        running_loss = 0
        for images, labels in loader_train:
            log_ps =model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(loader_train)}")
    #enregistrement du model dans un fichier 
    save_model(model)
    
    return model

def save_model(model):
    """enregistrement du model"""
    print("notre model: \n\n", model, '\n')
    print("dictionnaire d'etat: \n\n", model.state_dict().keys())
    torch.save(model.state_dict(), 'checkpoint.pth')