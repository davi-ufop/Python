### Programa muito simples para o treinamento do Reacher-v2 usando PPO
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos necessários
import gym                      ### Gym e MuJoCo constroem o ambiente pra RL 
import mujoco_py   
from stable_baselines3 import PPO   ### Modelo para o aprendizado de máquina
from time import sleep          ### Dá um tempo!

### Parâmetros e variáveis da simulação
pegou = 0       ### Contador de objetos capturados
tamanho = 0.01  ### Tamanho dos objetos, vide programa 01!

### Criando o ambiente: Representa um braço robótico que pega objetos em 2D
pegador = gym.make("Reacher-v2")
pegador.seed(123)    ### Estabalecendo a randomicidade 

### Definindo o modelo de aprendizagem: Q-Learn baseado na otimização da política
model = PPO("MlpPolicy", pegador, verbose=1)
model.learn(total_timesteps=5000)  ### Treinamento: 5000 passos, é suficiente?

### Testando o treinamento
obs = pegador.reset()   ### Primeira observação
for i in range(1000):   ### Tentando pegar 10 objetos em 1000 passos 
    ### Procedimentos para realizar o teste do modelo PPO
    acao, estado = model.predict(obs, deterministic=True) ### Previsão baseada na Q-table, define a ação e o estado
    obs, recompensa, pronto, info = pegador.step(acao)    ### Próximo passo, de acordo com a ação definida 
    pegador.render(mode='human', width=200, height=200)   ### Apresenta o pegador
    ### Preparando para a contagem de sucessos
    xtg = pegador.get_body_com("target")[0]         ### Coordenadas do alvo
    ytg = pegador.get_body_com("target")[1] 
    xfg = pegador.get_body_com("fingertip")[0]      ### Coordenadas do pegador
    yfg = pegador.get_body_com("fingertip")[1]
    dx = abs(xtg - xfg)    ### Parâmetros de controle, referentes a distância
    dy = abs(ytg - yfg)    ### entre o pegador e o alvo
    ### Contando os objetos capturados
    if (dx < tamanho and dy < tamanho):
      obs = pegador.reset()
      pegou += 1

### Pronto, acabou!
pegador.close()

### Resultados:
print("Foram capturados: ", pegou, " objetos.")

### FIM
