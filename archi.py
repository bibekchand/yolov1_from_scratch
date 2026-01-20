import torch
import torch.nn as nn
# architecture


class MyCNN(nn.Module):
    def __init__(self, S, B, C):
        self.S = S
        self.B = B
        self.C = C
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        convBlock4 = []
        for _ in range(4):
            convBlock4.append(nn.Conv2d(512, 256, kernel_size=1))
            convBlock4.append(nn.BatchNorm2d(256))
            convBlock4.append(nn.LeakyReLU(0.1, inplace=True))
            convBlock4.append(
                nn.Conv2d(256, 512, kernel_size=3, padding='same'))
            convBlock4.append(nn.BatchNorm2d(512))
            convBlock4.append(nn.LeakyReLU(0.1, inplace=True))
        convBlock4.append(nn.Conv2d(512, 512, kernel_size=1))
        convBlock4.append(nn.BatchNorm2d(512))
        convBlock4.append(nn.LeakyReLU(0.1, inplace=True))
        convBlock4.append(nn.Conv2d(512, 1024, kernel_size=3, padding='same'))
        convBlock4.append(nn.BatchNorm2d(1024))
        convBlock4.append(nn.LeakyReLU(0.1, inplace=True))
        convBlock4.append(nn.Conv2d(1024, 1024, kernel_size=3, padding='same'))
        convBlock4.append(nn.BatchNorm2d(1024))
        convBlock4.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.convBlock4 = nn.Sequential(*convBlock4)

        convBlock5 = []
        for _ in range(2):
            convBlock5.append(nn.Conv2d(1024, 512, kernel_size=1))
            convBlock5.append(nn.BatchNorm2d(512))
            convBlock5.append(nn.LeakyReLU(0.1, inplace=True))
            convBlock5.append(
                nn.Conv2d(512, 1024, kernel_size=3, padding='same'))
            convBlock5.append(nn.BatchNorm2d(1024))
            convBlock5.append(nn.LeakyReLU(0.1, inplace=True))
        convBlock5.append(nn.Conv2d(1024, 1024, kernel_size=3, padding='same'))
        convBlock5.append(nn.BatchNorm2d(1024))
        convBlock5.append(nn.LeakyReLU(0.1, inplace=True))
        convBlock5.append(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1))
        convBlock5.append(nn.BatchNorm2d(1024))
        convBlock5.append(nn.LeakyReLU(0.1, inplace=True))

        self.convBlock5 = nn.Sequential(*convBlock5)

        self.convBlock6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.ConnectedLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.S * self.S * (5 * self.B + self.C)),
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = self.convBlock5(x)
        x = self.convBlock6(x)
        x = self.ConnectedLayer(x)
        x = torch.reshape(
            x, (x.shape[0], self.S, self.S,  (5 * self.B + self.C)))
        return x


test_tensor = torch.randn([16, 3, 448, 448], dtype=torch.float, device='mps')
test_model = MyCNN(S=7, B=2, C=10)
test_model.to("mps")
y = test_model(test_tensor)
print(y.shape)
learning_rate = 0.01
epoch = 10
