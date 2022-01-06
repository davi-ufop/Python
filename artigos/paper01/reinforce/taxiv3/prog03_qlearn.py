### Programa para explorar a biblioteca gym de OpenAI
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Importando os módulos necessários
import numpy as np                  ## Numpy
from os import system               ## Pra apagar a tela
from time import sleep              ## Pra aguardar alguns segundos
from random import randint          ## Pra gerar estados aleatórios
import gym                          ## A BIBLIOTECA

### Limpando a tela no Linux, outro S.O., comente-me
system('clear')

### Parâmetros da simulação
K = 150000  ## Tamanho do treinamento, o aumente se houver BUG!
alfa = 0.1  ## Parâmetros do método Q-Learn, vide [1]
gama = 0.6
passos, penalidades = 0, 0  ## Contadores
Q = 50      ## Limite para o Q teste, não aumente!

### Criando o ambiente de simulação
taxi = gym.make("Taxi-v3").env

### Criando a tabela Q -> N° de estados = Linhas & N° de ações = colunas
q_tab = np.zeros([taxi.observation_space.n, taxi.action_space.n])

### Treinamento para preencher a tabela Q
for i in range(K):            ## Loop de K vezes
  ## Contando o treinamento
  print("Treino: ", i, "/", K)
  ## Escolhendo uma ação aleatória
  acao = randint(0,5)
  ## Apresentando o estado atual:
  estado = taxi.s             ## Determinando o estado atual 
  ## Passando para o próximo estado, usando a ação escolhida:
  proximo, recompensa, pronto, info = taxi.step(acao)
  ## Atualizando os valores Q (recompensas cumulativas)
  atual = q_tab[estado, acao]       ## Valor atual de Q
  maior = np.max(q_tab[proximo])    ## Maior Q para o próximo estado
  novoq = (1-alfa)*atual + alfa*(recompensa + gama*maior) ## [1] -> Q-Learn
  q_tab[estado, acao] = novoq       ## Atualiza (preenche) q_tab
  ## Contando penalidades
  if (recompensa == -10):
      penalidades += 1

### Testando o treinamento
system('clear')
print("Treino encerrado, vamos agora testar a nossa tabela Q!")
sleep(2)

### Condição de parada
pronto = False
while (pronto == False and passos<Q):
  ## Apresentando o passo
  print("Passo: ", passos, "/", Q)
  ## Definindo o estado (inicial) atual:
  estado = taxi.s
  print("Estado = ", estado)  ## Apresentando o estado
  print(taxi.render())        ## graficamente por
  sleep(0.3)                  ## 0.3 segundos
  ## Determinando a melhor ação
  acao = np.argmax(q_tab[estado])  ## Índice da Ação com o maior valor Q
  ## Passando para o próximo estado, usando a ação escolhida:
  proximo, recompensa, pronto, info = taxi.step(acao)
  system('clear')
  ## Destravando o taxi 
  if (proximo == estado):
    q_tab[estado, acao] = -100
  ## Contando os passos
  passos += 1

### Resultado
print("Sucesso alcançado em ", passos, " passos, com ", penalidades, " penalidades")

### FIM
### Referências:
### [1] https://en.wikipedia.org/wiki/Q-learning
