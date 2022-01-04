### Para plotar campos vetoriais do sistema dinâmico
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos necesários
import numpy as np
import matplotlib.pyplot as pl

### Parâmetros e Variáveis
Bx = -10            ## Botton de X, variável, será alterado
Tx = 10
By = -10
Ty = 10             ## Top de Y, variável, será alterado
Lx = 24             ## Parâmetros, não será variado   
Ly = 24             ## Tamanho da grade em Y, -> número de vetores

### Função de plotagem
def plot_campo(F, xran=[Bx, Tx], yran=[By, Ty], grid=[Lx, Ly]):
  ## Construindo o espaço de estados
  x = np.linspace(xran[0], xran[1], grid[0])
  y = np.linspace(yran[0], yran[1], grid[1])
  ## Vetores referentes as EDOs
  def dx_dt(X, Y, t=0):
    return map(eval, F)
  ## Construindo o Mesh para o campo vetorial
  X, Y = np.meshgrid(x, y)
  ## Definindo as direções dos vetores
  Dx, Dy = dx_dt(X, Y)
  ## Normalização dos vetores
  N = (np.hypot(Dx, Dy))   ## Norma 
  N[N == 0] = 1.0
  Dx = Dx/N                ## Normalizando
  Dy = Dy/N
  ## Plotando o campo vetorial com quiver()
  pl.quiver(X, Y, Dx, Dy, pivot='mid', color='black')
  pl.xlim(xran), pl.ylim(yran)
  pl.grid('on')

### FIM  (-> lê-se: indica)
