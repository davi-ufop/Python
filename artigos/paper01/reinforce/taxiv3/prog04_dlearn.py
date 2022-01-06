### Programa para treinar uma rede neural com o jogo do taxi (gym)
### Davi C. Neves - Ouro Preto, Brasil - Jan. 2022

### Importando módulos necessários
import numpy as np           ## Numpy
import gym                   ## GYM
from time import sleep       ## Sleep
from random import randint   ## RandInt
### Módulos da biblioteca Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
### Módulos da biblioteca keras-rl
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

### Criando o ambiente de simulação
taxi = gym.make("Taxi-v3")
taxi.render()

### Sementes dos processos randômicos
np.random.seed(123)
taxi.seed(123)

### Definindo a quantidade de ações e estados
num_acoes = taxi.action_space.n
num_estados = taxi.observation_space.n

### Resetando o sistema:
taxi.reset()
taxi.step(taxi.action_space.sample())[0]

### Criando a rede neural para o Deep Learn 
model = Sequential()    ## Função sequencial, vide [1]
model.add(Embedding(500, 10, input_length=1))  ### Empacotando 500 entradas em 10 neurônios
model.add(Reshape((10,)))                      ### Entradas ~ estados e Saídas ~ ações 
model.add(Dense(50, activation='relu'))        ### Camadas densas com 50 neurônios e
model.add(Dense(50, activation='relu'))        ### Função de ativação do tipo: relu
model.add(Dense(50, activation='relu'))
model.add(Dense(num_acoes, activation='linear'))  ### Útlima camada com apenas seis saidas (ações)
print(model.summary())                         ### Estrutura da rede: 500 ===> 10 -> 50 -> 50 -> 50 -> 6

### Treinamento da rede neural === DEEP LEARN
memory = SequentialMemory(limit=50000, window_length=1)   ### Vide [2]
policy = EpsGreedyQPolicy()     ### Aprendizagem gulosa, tipo Q-Learn
dqn = DQNAgent(model=model, nb_actions=num_acoes, memory=memory, nb_steps_warmup=500, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(taxi, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=100000)  ### Fit do Q-Learn

### Testando o aprendizado profundo
dqn.test(taxi, nb_episodes=5, visualize=True, nb_max_episode_steps=99)

### Salvando os pesos da rede treinada, serve pra ... 
#dqn.save_weights('dqn_{}_weights.h5f'.format("Taxi-v3"), overwrite=True)

### FIM
### Referência:
### [1] https://keras.io/api/models/sequential/
### [2] https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py
