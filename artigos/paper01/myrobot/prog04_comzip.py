### Programa para escolhermos a posição de três objetos
### Neste programa usamos a biblioteca 3, melhor elaborada, 
### pois usa o ZIP pra sincronizar as ações nos dois braços
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando módulos necessários
from mylib03 import *
from random import uniform

### Parâmetros e variáveis do processo
dteta = 0.01    ### Variação angular 
R = 2           ### Dimensão do ambiente 

### Pegando 3 objetos:
for i in range(3):
  ### Definindo a posição do objeto
  print("\nObjeto ", i+1)
  xo = eval(input("Digite a coordenada -1.5<X<1.5 do objeto: "))
  yo = eval(input("Digite a coordenada -1.5<Y<0 do objeto: "))
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
  da1 = acoes_junta(a1o, a1b, na1, dteta)
  da2 = acoes_junta(a2o, a2b, na2, dteta)

  ###### Realizando os movimentos para pegar o objeto
  caminho = "prog04/pegou{}.png".format(i+1)
  move_braco(da1, da2, a1b, a2b, R, xo, yo, caminho)

### FIM
