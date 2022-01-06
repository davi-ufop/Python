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
print("Iniciando o jogo. Atenção!")
print("Conte o número de passos e monte sua lista de ações.")
sleep(2)

### Apresentando o ambiente do jogo
taxi.render()

### Entrada de dados
print("Lembre-se: 0=desce, 1=sobe, 2=direita, 3=esquerda, 4=pega, 5=deixa")
nstep = eval(input("Digite a quantidade exata de ações: "))
lista = []
print("Agora anote a sequência de ações, antes de iniciar a digitação destas!")
for i in range(nstep):
  ac = int(input("Digite a próxima ação: "))
  lista.append(ac)
system('clear')

### Parâmetros de controle da simulação
penalidades = 0    ## Contadores de penalidades e passos
passos = 0 
proximo = taxi.s   ## Para o proximo estado

### Realizando o jogo:
for acao in lista:
  ### Mostrando o estado (inicial e posteriores):
  print("Estado = ", proximo)
  print(taxi.render())
  print("Ação escolhida: ", acao)
  sleep(0.3)
  system('clear')
  ### Próximo passo do taxi, usando a função step():
  proximo, recompensa, pronto, info = taxi.step(acao)
  ### Contando passos e penalidades
  if (recompensa == -10):
    penalidades += 1
  passos += 1  

### Resultados:
if (pronto == True):
    print("Você conseguiu, parabéns!")
else:
    print("O jogo acabou e você não conseguiu.")
### Contagem
print("Foram realizados ", passos, " passos.")
print("Com ", penalidades, " penalidades.")

### FIM
