import os
import random
import re
import time
import uuid
from typing import List, Tuple

import numpy as npy
from PIL import Image
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from torchmetrics.classification import Accuracy, Precision, Recall


NAMESPACE_TEST = uuid.uuid3(uuid.NAMESPACE_DNS, "test")

folder = "../diffuser/inpaint"


class CityScapesDataset(Dataset):
    task: str

    def __init__(self, path: str, task: str, transform=None):
        self.task = task
        self.images, self.labels, self.evals, self.inpaints = self.get_images(path)
        self.transform = transform

    def __getitem__(self, item) -> Tuple[npy.ndarray, int]:
        path_img, label = self.evals[item // 3] if task == "TEST" else self.images[item // 3] if task == "TRAIN" else self.inpaints[item // 3]
        cp = lambda index: (index * 112, 0, index * 112 + 224, 224)
        img = npy.array(Image.open(path_img).convert("RGB").resize((448, 224), Image.LANCZOS).crop(cp(item % 3))).astype(npy.float32) * 2.0 / 255.0 - 1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if task == "TEST":
            return len(self.evals) * 3
        if task == "TRAIN":
            return len(self.images) * 3
        return len(self.inpaints) * 3

    def get_images(self, path: str) -> Tuple[List, List]:
        images = []
        evals = []
        inpaints = []

        pattern = r'^[a-z]+$'

        labels = sorted(list(filter(
            lambda x: re.search(pattern, x) != None,
            os.listdir(path)
        )))

        for l in labels:
            p2 = os.path.join(path, l)
            images_train = list(filter(
                lambda x: x.endswith(".png"),
                os.listdir(p2)
            ))
            images_eval = random.sample(images_train, 5)
            for im in images_train:
                if im in images_eval:
                    continue
                path_img = os.path.join(p2, im)
                images.append((path_img, labels.index(l)))
                uid = uuid.uuid3(NAMESPACE_TEST, im)
                inpaints.append((folder + "/%s.png"%uid, labels.index(l)))
            for im in images_eval:
                path_img = os.path.join(p2, im)
                evals.append((path_img, labels.index(l)))

        return images, labels, evals, inpaints
    
    def get_num_classes(self) -> int:
        return len(self.labels)

task = "TEST_INPAINT"

device = torch.device("cuda:0")

random.seed(42)
dataset = CityScapesDataset("dataset", task, transform=transforms.ToTensor())
print("Loaded %d images."%len(dataset))

print("Task:", task)

lr_multiplier = 0.1

if task == "TRAIN":
    dataIter = DataLoader(dataset, batch_size=200, shuffle=True)
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(512 * 4, dataset.get_num_classes())
    model = torch.load("models/metrics/cityscapes_classifier.pth")
    model = model.to(device)
    optimizer = Adam([
        {'params': model.fc.parameters(), 'lr': 1e-3 * lr_multiplier, 'weight_decay': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 2e-5 * lr_multiplier, 'weight_decay': 2e-6},
        {'params': model.layer3.parameters(), 'lr': 1.2e-5 * lr_multiplier, 'weight_decay': 1.2e-6},
        {'params': model.layer2.parameters(), 'lr': 1e-5 * lr_multiplier, 'weight_decay': 1e-6},
        {'params': model.layer1.parameters(), 'lr': 1e-5 * lr_multiplier, 'weight_decay': 1e-6},
        {'params': model.bn1.parameters(), 'lr': 1e-5 * lr_multiplier, 'weight_decay': 1e-6},
        {'params': model.conv1.parameters(), 'lr': 1e-5 * lr_multiplier, 'weight_decay': 1e-6}
    ])
    loss_function = nn.CrossEntropyLoss()

    print("Start training.")
    model.train()
    for epoch in range(8):
        train_loss = 0
        for data, label in dataIter:
            pred = model(data.to(device))

            optimizer.zero_grad()
            loss = loss_function(pred, label.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()

            continue
        train_loss /= len(dataIter.dataset)
        print("Epoch = %d, Loss ="%epoch, train_loss)

    torch.save(model, "models/metrics/cityscapes_classifier.pth")
    torch.save(model.state_dict(), "models/metrics/cityscapes_classifier_parameter.pth")
elif task.startswith("TEST"):
    acc1 = Accuracy(num_classes=dataset.get_num_classes(), top_k=1).to(device)
    acc2 = Accuracy(num_classes=dataset.get_num_classes(), top_k=2).to(device)
    acc3 = Accuracy(num_classes=dataset.get_num_classes(), top_k=3).to(device)
    acc4 = Accuracy(num_classes=dataset.get_num_classes(), top_k=4).to(device)
    dataIter = DataLoader(dataset, batch_size=200, shuffle=False)
    model = torch.load("models/metrics/cityscapes_classifier.pth")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data, label in dataIter:
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            # for i in range(data.shape[0]):
            #     if random.randint(0, 15) == 0:
            #         print(pred[i], label[i])
            acc1.update(pred, label)
            acc2.update(pred, label)
            acc3.update(pred, label)
            acc4.update(pred, label)
        print("Acc@1: " + str(acc1.compute().detach().cpu().item()))
        print("Acc@2: " + str(acc2.compute().detach().cpu().item()))
        print("Acc@3: " + str(acc3.compute().detach().cpu().item()))
        print("Acc@4: " + str(acc4.compute().detach().cpu().item()))
elif task == "LABELS":
    for l in dataset.labels:
        print(l, dataset.labels.index(l))
