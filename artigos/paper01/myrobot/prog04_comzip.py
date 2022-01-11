### Programa para escolhermos a posição de três objetos
### Neste programa usamos a biblioteca 3, melhor elaborada, 
### pois usa o ZIP pra sincronizar as ações nos dois braços
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando módulos necessários
from mylib03 import *
from random import uniform
import warnings
warnings.filterwarnings('ignore')

### Parâmetros e variáveis do processo
dteta = 0.01      ### Variação angular 
R = 2             ### Dimensões do ambiente 
B = R/np.sqrt(2)  ### Alcance máximo do braço
tamanho = 0.04    ### Dimensão do objeto

### Total de estados
print("Total de estados = ", (2.2*(R*R)/2)//tamanho)

### Pegando 3 objetos:
for i in range(3):
  ### Definindo a posição do objeto, na mesa!
  print("\nObjeto ", i+1)
  xo = round(tamanho*(uniform(-B, B)//tamanho), 2)
  yo = round(tamanho*(uniform(-B, 0)//tamanho), 2)
  ### Definindo os ângulos alvos
  [a1o, a2o] = angulos_ponto(xo, yo, 1, 1)

  ### Definindo a posição do braço
  xb = uniform(-R, R)
  yb = uniform(-R, 0)
  ### Definindo o estado do sistema
  [a1b, a2b] = angulos_ponto(xb, yb, 1, 1)

  ### Determinando o número de ações
  na1 = int( abs(a1b-a1o)//dteta )
  na2 = int( abs(a2b-a2o)//dteta )
  nat = na1 + na2
  print("Serão realizadas ", nat, " ações, divididas assim: ", [na1, na2])

  ###### Construindo as ações - MUITO IMPORTANTE
  ### Listas para as ações nas juntas do braço
  da1 = acoes_listas(a1o, a1b, dteta)
  da2 = acoes_listas(a2o, a2b, dteta)

  ###### Realizando os movimentos para pegar o objeto
  caminho = "prog04/pegou{}.png".format(i+1)
  move_braco(xo, yo, a1b, a2b, da1, da2, R, caminho, tamanho)

### FIM
