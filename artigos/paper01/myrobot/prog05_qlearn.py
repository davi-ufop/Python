"""
### Programa que o robô vai usar uma q-table para aprender a pegar três objetos
### Ações:
A0 -> -1, -1     
A1 -> -1, +1     
A2 -> -1,  0     
A3 -> +1, -1    
...
A8 ->  0,  0  (Pegou o objeto) 

### Estados:
  Estado 0 = [0, -2]
  Estado 1 ->   Estado 0 + A0
  Estado 2 ->   Estado 1 + A0 
  Estado 3 ->   Estado 2 + A0 

### Davi Neves - Ouro Preto, Brasil - Jan., 2022 """

### Importando módulos necessários
from mylib04 import *
from random import seed, randint, uniform, choice
import warnings
warnings.filterwarnings('ignore')

### Parâmetros e variáveis do processo
R = 2             ### Dimensões do ambiente 
B = R/np.sqrt(2)  ### Alcance máximo do braço
dteta = 0.05      ### Tamanho da variação angular, valor adequado! Não aletere!
tamanho = 0.05    ### Dimensão do objeto
P = 2             ### Precisão das medidas
L1, L2 = R/2, R/2

############ INICIANDO A DIVERSÃO
seed(888)
### Pegando 3 objetos:
for i in range(3):
  ########## PARÂMETROS E VARIÁEIS INICIAIS
  ### Definindo a posição do objeto, na mesa!
  print("\nObjeto ", i+1)
  xo = round(tamanho*(uniform(-B, B)//tamanho), P)
  yo = round(tamanho*(uniform(-B, 0)//tamanho), P)
  print("xo = ", xo, "yo = ", yo)
  ### Definindo a posição do braço
  xb = round(uniform(-B, B), P)
  yb = round(uniform(-B, 0), P)
  print("xb = ", xb, "yb = ", yb)
  
  ########## LISTAS DE AÇÕES -> CARTESIANAS E ANGULARES
  ### CARTESIANAS:
  ### Lista de pontos cartesianos
  lpx =  acoes_listas(xo, xb, dteta)
  lpy =  acoes_listas(yo, yb, dteta)
  lpx, lpy = igualistas(lpx, lpy)
  ### ANGULARES:
  ### Determinando os ângulos
  [o1, o2] = angulos_ponto(xo, yo, L1, L2)
  [a1, a2] = angulos_ponto(xb, yb, L1, L2)
  ### Lista dos ângulos das juntas
  la1 =  acoes_listas(o1, a1, dteta)
  la2 =  acoes_listas(o2, a2, dteta)
  la1, la2 = igualistas(la1, la2)
 
  ########## ESTADOS DO SISTEMA -> CARTESIANOS E ANGULARES
  ### Estados do sistema em X e Y
  vxp, vyp = varia_estados(xb, yb, lpx, lpy)
  ### Estados do sistema em ângulos
  va1, va2 = varia_estados(a1, a2, la1, la2)
  ### Trajetória do pêndulo
  vxa = L1*(np.sin(va1)) + L2*(np.sin(va2))
  vya = L1*(np.cos(va1)) + L2*(np.cos(va2))
  ### Plotando as duas trajetórias
  salvem = "prog05/trajetoria{}.png".format(i+1)
  plot_trajetorias(vxa, vya, vxp, vyp, salvem)
  ###### Realizando os movimentos para pegar o objeto
  caminho = "prog05/pegou{}.png".format(i+1)
  move_braco(xo, yo, a1, a2, la1, la2, R, caminho, tamanho)

#####################################################################
############ INICIANDO O TREINAMENTO
### Contando estados e ações:
AN = 9           
### Precisão dos estados para o treino:
dd = 0.04   ### Se alterar esse valor, comente a linha 113
### Confirmando os estados contínuos -> angulares
saved = "prog05/estados.png"
EN, TAG, TXY = plot_estados(R, dd, saved)

### Verificando se todos os estados estão contemplados
Bom, Ruim = completo(R, dd, TXY)
print("\nDeu Ruim = ", Ruim)
print("Deu Bom = ", Bom)

### Criando a tabela Q de acordo com nossa introdução
qtab = np.zeros([EN, AN])
print("Número de estados = ", len(qtab))

### Definindo o estado do objeto
seed(888)
xo = round(dd*(uniform(-B, B)//dd), P)
yo = round(dd*(uniform(-B, 0)//dd), P)
### Descobrindo o estado do objeto
print("Objeto: x = ", xo, " y = ", yo)
IO = np.where(np.all(TXY == (xo, yo), axis=1))[0][0]
print("Estado do objeto = ", IO)

### Dando a recompensa para achá-lo
qtab[IO, 8] = 20

### Realizando os treinos
print("\nTreinando -> Q-Table:")
K = 2000000    ### Se alterar essa linha, comente a linha 113
#qtab = treino(qtab, K, TXY, xo, yo, dd, B)                ### Para refazer a simulação comente
qtab = np.genfromtxt('prog05/qtable.csv', delimiter=',')   ### a linha acima e ative essa
#np.savetxt("prog05/qtable.csv", qtab, delimiter=',')      ### Comente esta linha também 

### Apresentando a Q-table atualizada
print(qtab[IO-3:IO+3])

### Testando o aprendizado
semente = eval(input("Digite a semente randômica: "))
seed(semente)    
###lista = []
###for k in tqdm(range(100, 499)):
###  seed(k)
###  tres = 0
for i in range(3):
  xb = round(dd*(uniform(-B, B)//dd), P)           ### Estado inicial
  yb = round(dd*(uniform(-B, 0)//dd), P)
  saveme = "prog05/inteligente{}.png".format(i+1)  ### Resultados
  le = testeQ(qtab, TXY, xb, yb, xo, yo, dd, saveme, i+1)
  ### Resultado do teste
  print("\nResultado:")
  print("Objeto alcançado em ", len(le), " passos.\nEstes:\n", le)
###    if (len(le) < 60):
###      tres += 1
###  if (tres == 3):
###    print(k)
###    lista.append(k)
### LISTA de Sementes Testadas, um teste do Teste!
lista = [109, 117, 127, 131, 154, 156, 162, 166, 183, 186, 192, 197, 198, 199, 208, 213, 214, 215, 221, 223, 224, 226, 232, 235, 239, 250, 257, 262, 267, 279, 286, 290, 293, 294, 301, 306, 308, 309, 313, 314, 315, 317, 320, 337, 340, 355, 357, 375, 381, 382, 393, 396, 399, 406, 412, 418, 428, 430, 434, 436, 440, 442, 458, 467, 480, 482, 483, 485, 486, 498]

### Eficiência
print("\nEficiência do processo = ", round(len(lista)/399, 2))

### FIM
