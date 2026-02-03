from archi import MyCNN
import torch
from torch.utils.data import Dataset, DataLoader
from loss import loss
model = MyCNN(S=7, B=2, C=20)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x_test_tensor = torch.randn(
    [16, 3, 448, 448], dtype=torch.float)
y_test_tensor = torch.randn([16, 7, 7, 25], dtype=torch.float)
dataset = CustomDataset(x_test_tensor, y_test_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


def train():
    model.train()
    for epoch in range(epochs):
        for batch_features, batch_labels in dataloader:
            y_pred = model(batch_features)
            loss_value = loss(y_pred, batch_labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        print(epoch, loss_value.item())
