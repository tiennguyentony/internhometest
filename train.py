import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
from flask_socketio import SocketIO

import time


def calculate_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro")


def calculate_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")


class MNISTModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.layers(x)


def construct(lr=None, bs=None, dropout_p=None):
    model = MNISTModel(dropout_p)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        datasets.MNIST(
            "../data", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=bs,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "../data", train=False, download=True, transform=transforms.ToTensor()
        ),
        batch_size=bs,
    )
    return model, train_loader, test_loader, loss_fn, optimizer


def train(model, train_loader, loss_fn, optimizer, callback=None, epochs=None):
    for epoch in range(epochs):

        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):

            outputs = model(images.reshape(-1, 28 * 28))
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if callback is not None:
            callback(epoch + 1, epochs)

        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")

    print("Finished training")

    return model


def test(model, test_loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.reshape(-1, 28 * 28))
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = calculate_precision(all_labels, all_predictions)
    recall = calculate_recall(all_labels, all_predictions)

    print(f"Test accuracy: {accuracy*100}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    return accuracy, precision, recall
