#### Ajuste do Programa 05, usando um dataset local, sem download
#### Adaptado por Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos necessários
import numpy as np
import pylab as pl
from tqdm import tqdm
### PyTorch
import torch
from torch.utils.data import DataLoader as DL
import torchvision
from torchvision.datasets import ImageFolder as IF
from torchvision import transforms

# Transforms images to a PyTorch Tensor
tensor_transform = transforms.ToTensor()
  
# DataLoader is used to load the dataset for training
loader = DL(dataset=IF("data/numbers/", transform=tensor_transform), batch_size=64, shuffle=True)

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Model Initialization
model = AE()
  
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
  
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)

### Train
epochs = 5
outputs = []
losses = []
for epoch in tqdm(range(epochs)):
  for (image, _) in loader:
    # Reshaping the image to (-1, 784)
    image = image.reshape(-1, 28*28)
    # Output of Autoencoder
    reconstructed = model(image)
    # Calculating the loss function
    loss = loss_function(reconstructed, image)
    # The gradients are set to zero,
    # the the gradient is computed and stored.
    # .step() performs parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Storing the losses in a list for plotting
    losses.append(loss.item())
  outputs.append((epochs, image, reconstructed))
  
# Plotting the last 100 values
#pl.plot(losses[-100:])
#pl.title('Loss Function')
#pl.xlabel('Iterations')
#pl.ylabel('Loss')
#pl.show()
#pl.savefig("data/numbers/imgs/fig01_loss_function.png", dpi=250)
#pl.clf()

# Results
for img in zip(image[0:2], reconstructed[0:2]):
  item1 = img[0].reshape(-1, 28, 28)
  item2 = img[1].reshape(-1, 28, 28)
  fig, ax =  pl.subplots(1, 2)
#  ax[0].imshow(item1[0])
#  ax[1].imshow(item2[0].detach().numpy())
#  pl.show()
#  pl.clf()

### FIM
