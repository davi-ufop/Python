### Programa para realizar a convolução das imagens
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos
import numpy as np
import pylab as pl
from torch import flatten, nn
from torchvision.transforms import transforms

### Pegando e mostrando a figura 23
imagem = np.genfromtxt('data/out_023.csv', delimiter=',')
pl.imshow(imagem)
pl.show()
pl.clf()

print("As dimensões dos tensores:")
###### Operações de convolução
### Transformando em tensor
transforme = transforms.ToTensor()   ### Módulo pra tranformar em tensor
tensor = transforme(imagem)          ### Tranformando e tensor
print("Tensor inicial = ", tensor.size())
### Realizando o polimento
polimento = nn.MaxPool2d(kernel_size=4, stride=2, return_indices=True)
tensor, inds = polimento(tensor)
print("Tensor polido = ", tensor.size())
dim = tensor.size()[1]
### Realizando o achatamento: 2D -> 1D
tensor = flatten(tensor, start_dim=1)  
print("Tensor achatado = ", tensor.size())

###### Operações de reconvolução
### Inflação do tensor 
tensor = tensor.view(1, dim, dim)
print("Tensor inflado = ", tensor.size())
### Despolindo o tensor
despolimento = nn.MaxUnpool2d(kernel_size=4, stride=2)
tensor = despolimento(tensor, inds)
print("Tensor reconstruido = ", tensor.size())

### Reconstruindo a figura
tensor = tensor.view(128, 128)    ### Forma original 
figura = tensor.detach().numpy()  ### Converte para numpy
### Apresentando a figura reconstruida
pl.imshow(figura)
pl.show()
pl.clf()

print("Nossa rede deve ter ", dim*dim, " entradas.")

### FIM
