### Programa que está implementado um autoencoder usando 
### imagens de celebridades, retirado do Kaggle, vide [1]
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos necessários
import os                             # Sistema operacional 
import numpy as np                    # Numérica
import pylab as pl                    # Gráfica
import imageio                        # Imagens
from tqdm import tqdm                 # Barra de progresso

### Bibliotecas e módulos referentes ao PyTorch 
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms

### Evitando Warnings irrelevantes
import warnings
warnings.filterwarnings('ignore')

### Diretórios do dataset de imagens
IMG_DIR = "data/kaggle/imgs/"
IMGDIR = "data/kaggle/"

### Testando o dataset
pl.figure(figsize=(15,10))
for i in range(6):
    pl.subplot(2,3,i+1)
    choose_img = np.random.choice(os.listdir(IMG_DIR))
    image_path = os.path.join(IMG_DIR,choose_img)
    image = imageio.imread(image_path)
    pl.imshow(image)
pl.show()

### Classe para contruir AutoEncoders
class Autoencoders(nn.Module):
    def __init__(self):
        super().__init__()
        ### Encoder
        self.conv1 = nn.Conv2d(3,64,5)
        self.maxpool = nn.MaxPool2d(2,return_indices=True)
        self.conv2 = nn.Conv2d(64,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        ### Decoder
        self.deconv1 = nn.ConvTranspose2d(128,64,5)
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(64,64,5)
        self.deconv3 = nn.ConvTranspose2d(64,3,5)
    def forward(self,x):
        x = self.conv1(x)
        x,ind1 = self.maxpool(x)
        x = self.conv2(x)
        x,ind2 = self.maxpool(x)
        x = self.conv3(x)
        x = self.deconv1(x)
        x = self.unpool(x,ind2)
        x = self.deconv2(x)
        x = self.unpool(x,ind1)
        x = self.deconv3(x)
        return x

### Preparando os dados para o treinamento
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = torchvision.transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])#, normalize ])
train_dataloader = torch.utils.data.DataLoader(dataset=torchvision.datasets.ImageFolder(IMGDIR, transform=train_transform), shuffle=True, batch_size=32, num_workers=0)

### Preparando o código para rodar em GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
model = Autoencoders().to(device)

### Criando a função perda e o otimizador
criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

### Realizando o treinamento
n_epochs = 1  ### Tempo para n_epochs=1: 130s
for epoch in tqdm(range(n_epochs)):
    model.train()
    iteration = 0
    for data,_ in train_dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data)
        loss = criterian(output,data)
        loss.backward()
        optimizer.step()
        if iteration%1000 == 0:
            print(f'iteration: {iteration} , loss : {loss.item()}')
    print(f'epoch: {epoch} loss: {loss.item()}')

### Salvando e usando o modelo
torch.save(model.state_dict(),'kaggle.h5')
model1 = Autoencoders()
model1.load_state_dict(torch.load('kaggle.h5'))
model1.eval()

### Pegando a 17ª figura
pega = 0
for data,_ in train_dataloader:
    pega += 1
    if (pega > 16):
        break
### Previsao para esta figura
pred_img = model1(data)
pred_img = pred_img.detach().numpy()
pred_img = pred_img.reshape(32,224,224,3)
pl.imshow(pred_img[0])
pl.show()
### Conferindo a resposta
new_data = data.reshape(32,224,224,3)
pl.imshow(new_data[0])
pl.show()

### Limpando a pasta
os.system('rm *.h5')
### FIM
### Referências:
### [1] https://www.kaggle.com/residentmario/autoencoders
