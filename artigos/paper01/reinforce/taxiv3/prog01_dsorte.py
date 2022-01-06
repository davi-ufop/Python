### Programa para explorar a biblioteca gym de OpenAI
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Importando os módulos necessários
from os import system               ## Pra apagar a tela
from time import sleep              ## Pra vc ver apagando a tela
from random import randint          ## Pra gerar estados aleatórios
import gym                          ## A BIBLIOTECA

### Criando o amebiente de simulação
taxi = gym.make("Taxi-v3").env

### Controle da aleatoriedade
taxi.seed(111)

######### Iniciando a simulação
print("Iniciando a simulação random-walk!")
sleep(2)

### Parâmetros de controle da simulação
pronto = False     ## Parada para a simulação
penalidades = 0    ## Contadores de penalidades e passos
passos = 0 
proximo = taxi.s   ## Para o proximo estado

### Realizando a simulação até a tarefa estar pronta!
while (pronto == False):
  ### Mostrando o estado (inicial e posteriores):
  print("Estado = ", proximo)
  print(taxi.render())
  sleep(0.2)
  system('cls||clear')
  ### Sorteando uma ação possível:
  acao = taxi.action_space.sample()
  print("Ação escolhida: ", acao)
  ### Próximo passo do taxi, usando a função step():
  proximo, recompensa, pronto, info = taxi.step(acao)
  ### Contando passos e penalidades
  if (recompensa == -10):
    penalidades += 1
  passos += 1  
  ### Ajudando a sorte!
  pegue = taxi.P[taxi.s][4][0][2]   ## Variável pra pegar o cliente
  deixe = taxi.P[taxi.s][5][0][2]   ## Variável pra deixar o cliente
  ### Condição pra pegar o cliente e 5 ações pra sair do ponto
  if (pegue == -1):
    proximo, recompensa, pronto, info = taxi.step(4)  ## Pega o cliente
    for i in range(5):  ## Sai do ponto
      proximo, recompensa, pronto, info = taxi.step(randint(0,3))
  ### Condição pra deixar o o cliente e acabar a tarefa
  if (deixe == 20):
      proximo, recompensa, pronto, info = taxi.step(5)  ## Pronto!

### Resultados do sucesso!?
print("Foram realizados ", passos, " passos.")
print("Cumprindo ", penalidades, " penalidades.")

### FIM
