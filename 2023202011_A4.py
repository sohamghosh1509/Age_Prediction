import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, RandomResizedCrop
import torch.optim as optim
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)
    
train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)  # Output single value for age prediction

def train(model, loader, optimizer, criterion, device, num_epochs=40):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))  # Convert labels to 2D tensor for MSE loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(loader)}')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train(model, train_loader, optimizer, criterion, device)

###### SUBMISSION CSV FILE #####

@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

preds = predict(test_loader, model)

submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
submit['age'] = preds
submit.head()

submit.to_csv('submission.csv',index=False)