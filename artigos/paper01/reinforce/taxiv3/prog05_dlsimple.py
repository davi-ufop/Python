### Copiado de: https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/
### Ajustado por Davi Neves, em Ouro Preto - Brasil, no mês de Jan. de 2022 

### Importando os bibliotecas e módulos necessários
import numpy as np      ### Cálculo numérico
import gym              ### Ambiente para o reinforcemente learning 
### Biblioteca Keras:
from keras.models import Sequential     ## Tipo de modelo da rede neural
from keras.layers import Dense, Activation, Embedding, Reshape   ## Estruturas das redes neurais
from keras.optimizers import Adam   ## Modelo pra otimização dos pesos
### Biblioteca keras-rl:
from rl.agents.dqn import DQNAgent  ## Treinamento Q do agente 
from rl.policy import EpsGreedyQPolicy  ## Política gulosa
from rl.memory import SequentialMemory  ## Armazenamento de memória sequencial

### Criando o ambiente de simulação
taxi = gym.make("Taxi-v3")
taxi.render()

### Parâmetros e sementes do processo
np.random.seed(123)
taxi.seed(123)
num_acoes = taxi.action_space.n   ## Quantidade possíveis de ações

### Criando a estrutura da rede neural para o Deep Learn
model = Sequential()    ## Tipo do modelo
model.add(Embedding(500, 10, input_length=1))  ## Preparando a entrada dos dados: 500 estados em
model.add(Reshape((10,)))                      ## 10 neurônios pra inserir na camada oculta
### Camada oculta com 50 neurônios e função de ativação do tipo: relu
model.add(Dense(50))                         
model.add(Activation('relu'))
### Camada de saída, com função de ativação do tipo: linear
model.add(Dense(num_acoes))     ## 6 neurônios, um pra cada ação: 0, 1, 2, 3, 4 e 5
model.add(Activation('linear'))  
### Estrutura da rede montada: 500 ==> 10 -> 50 -> 6
print(model.summary())

### Treinamento do modelo usando a biblioteca keras-rl, vide [1]
policy = EpsGreedyQPolicy()  ## Definindo o tipo da política de aprendizagem
memory = SequentialMemory(limit=50000, window_length=1)  ## Definindo a memória
dqn = DQNAgent(model=model, nb_actions=num_acoes, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])  ## Compilando o modelo de aprendizado Q-Learn
dqn.fit(taxi, nb_steps=5000, visualize=True, verbose=2)  ## Treinando!!!

### Testando a rede treinada:
dqn.test(taxi, nb_episodes=5, visualize=True)  

### FIM
### [1] https://github.com/keras-rl/keras-rl
