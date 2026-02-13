from archi import MyCNN
from loss import loss

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# -----------------------
# MODEL
# -----------------------
S = 7
B = 2
C = 19

model = MyCNN(S=S, B=B, C=C)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# -----------------------
# DATASET
# -----------------------


class YOLOv1Dataset(Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=20):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.images = os.listdir(img_dir)

        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # ----- IMAGE -----
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image).float()

        # ----- LABEL -----
        label_path = os.path.join(
            self.label_dir,
            img_name.replace(".jpg", ".txt")
        )

        target = torch.zeros((self.S, self.S, self.C + 5*self.B))

        boxes = np.loadtxt(label_path, ndmin=2)

        for box in boxes:
            class_label, x, y, w, h = box
            class_label = int(class_label)

            i = int(self.S * y)
            j = int(self.S * x)

            x_cell = self.S * x - j
            y_cell = self.S * y - i
            w_cell = w * self.S
            h_cell = h * self.S

            if target[i, j, self.C] == 0:
                target[i, j, self.C] = 1
                target[i, j, class_label] = 1
                target[i, j, self.C+1:self.C+5] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell]
                )

        return image, target


# -----------------------
# DATALOADER
# -----------------------
dataset = YOLOv1Dataset(
    img_dir="dataset/train/images",
    label_dir="dataset/train/labels",
    S=S,
    B=B,
    C=C
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True
)

# -----------------------
# TRAINING SETUP
# -----------------------
epochs = 10
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4
)

# -----------------------
# TRAIN LOOP
# -----------------------


def train():
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch}")
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss_value = loss(predictions, targets)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.4f}")


train()
