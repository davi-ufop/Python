### Programa para resolver o problema similar ao braço robótico
### usando o ambiente Reacher-v2 de GYM
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### módulo es bibliotecas
from mylib09 import *
import os                   # Sistema operacional
import numpy as np          # Numérica
import pylab as pl          # Gráfica
from scipy.integrate import odeint  # EDOs
from tqdm import tqdm       # Barra de progresso
from random import uniform  # Aleatório
from imageio import imread  # Imagens
import gym                  # Ambiente de simulação RL
from stable_baselines3 import SAC, DQN  # Modelos de RL
from stable_baselines3.common.evaluation import evaluate_policy # Avalia Pi(s/a)

### Limpeza pertinente
os.system('rm -rf prog10/imgs/*.png')

"""
###### BANDO DE DADOS REFERENTE AO SISTEMA DINÂMICO
### Listas cuja a coordenada y do pêndulo foi < -1.7, ~ completude do espaço abrangido
lq1 = [0.13,0.54,0.7,0.69,0.14,-0.1,-0.65,0.57,-0.36,0.68,0.86,-0.51,0.99,0.09,0.73,-0.61,0.97]
lq2 = [0.49,-0.17,0.92,-0.2,1.31,0.61,-0.35,0.62,0.04,1.26,0.73,-0.22,-1.47,0.31,0.77,1.07,0.74]
### Pêndulo Duplo - 17 trajetórias com com 50 mil passos cada
LL = 500            # Limite pra t, deve ser alto
dL = 0.01           # Precisão dos outros códigos
Passos = LL/dL      # 50000 passos
NTrajs = len(lq1)   # 17 trajetórias
LAQ = []            # Lista pra registro
for xq in zip(lq1, lq2):
  t = np.arange(0, LL, dL)         # Variação temporal
  x0 = [xq[0], 0, xq[1], 2]        # Condição inicial
  x = odeint(dpendulo, x0, t)      # Solução numérica com precisão dL
  LAQ.append(np.transpose(x))      # Registrando
### Salvando pra não rodar mais  
NL = int(NTrajs*Passos)            # Número de linhas da tabela 850000
VAQ = np.array(LAQ).reshape(NL, 4) # Forma adequada
np.savetxt("prog10/data/dados_caos.csv", VAQ, delimiter=",") # Salvando em CSV
"""

###### APRENDIZADO POR REFORÇO - SAC
### Criando o ambiente de simulação -> Reacher-v2 emula um braço robótico
### que deve pegar o objeto, a saída de cada passo de simulação é:
### [cos1, cos2, sin1, sin2, xo, yo, vx_b, vy_b, dx_bo, dy_bo, dz_bo]*
pendulo = gym.make("Reacher-v2")

### Parâmetro do modelo:
dt = 0.01   # github.com/openai/gym/blob/master/gym/envs/mujoco/assets/reacher.xml

### Instância do modelo para simulação -> [1] 
### SAC é pertinnte em sistema com espaço de estados contínuos
### Caso o espaço de estados for discreto use DQN()
modelo1 = SAC('MlpPolicy', pendulo, verbose=1)

### Treinando o agente!!! 200000 treinos 
#modelo1.learn(total_timesteps=int(2e5))
### Salvando o modelo treinado!!!
#modelo1.save("prog10/data/modelo10")
### As duas etapas acima gastam 17 MINUTOS, ative
### os comandos acima na primeira simulação! 

### Metodologia para comparar dois modelos, basta não deletar
del modelo1  # Deletando o modelo para usar arquivo acima
modelo2 = SAC.load("prog10/data/modelo10", env=pendulo) # Carregando

### Estatísticas de recompensas (30 episódios/testes):
media, desvio = evaluate_policy(modelo2, modelo2.get_env(), n_eval_episodes=10)
np.savetxt("prog10/data/stats.csv", np.array([media, desvio]), delimiter=",")

### O verdadeiro teste!
LAG, LXY, LMo = [], [], []    # Listas de registros
obs = pendulo.reset()         # Reiniciando o sistema
ko = 0                  # Contador de sucesso: objetivo é pegar 3 objetos
for i in range(3000):   # 1000 testes
  alvo = [obs[4], obs[5]]
  acao, estado = modelo2.predict(obs, deterministic=True)      # Previsão
  obs, recompensa, pronto, info = pendulo.step(acao)           # Passo
  pendulo.render()                                             # Renderiza
  ### Salvando informações importantes
  Q1 = np.amax([np.arccos(obs[0]), np.arcsin(obs[2])])
  Q2 = np.amax([np.arccos(obs[1]), np.arcsin(obs[3])])
  LAG.append([Q1, Q2])   # Ângulos das juntas 
  if pronto:             # Condição de parada do teste
    LXY.append(alvo)     # Posições do Alvo 
    ### Parâmetros para plotagem
    N = len(LAG)-1
    VAG = np.array(LAG)
    Q1 = VAG[0,0]        # Primeiros ângulos das juntas do robô
    Q2 = VAG[0,1]
    ### Ângulo e velocidade da primeira junta
    teta = VAG[:, 0]
    omega1 = np.diff(teta, n=1)/dt
    teta1 = teta[0:N]
    ### Ângulo e velocidade da segunda junta
    teta = VAG[:, 1]
    omega2 = np.diff(teta, n=1)/dt
    teta2 = teta[0:N]
    ### Buscando soluções referentes ao sistema de equações diferenciais
    x0 = [Q1, Q2, teta1[N-1], teta2[N-1]]   # Condição Inicial
    x0 = np.round(np.array(x0), 2)          # Precisão necessária
    q1i, q2i, q1f, q2f = x0                 # Formato adequado pra busca
    QL = busca_listas(q1i, q2i, q1f, q2f)   # Busca listas
    if (len(QL) > 1):
      LMo.append(ko+1)
      plot_traj_caos(QL, "prog10/imgs/trajcaos{:02d}.png".format(ko+1))
    else:
      pass
    ### Solução referente ao procedimento de um Motor de passo - Eng. Mecânica
    La1, La2 = liang(teta1[0], teta2[0], teta1[N-1], teta2[N-1])
    ### Coordenadas X e Y
    Xm, Ym = np.sin(La1)+np.sin(La2), np.cos(La1)+np.cos(La2)
    Xs, Ys = np.sin(teta1)+np.sin(teta2), np.cos(teta1)+np.cos(teta2)
    ### Salvando para análise
    np.savetxt("prog10/data/motor_{:02d}.csv".format(ko+1), np.hstack((Xm, Ym)), delimiter=",")
    np.savetxt("prog10/data/prevs_{:02d}.csv".format(ko+1), np.hstack((Xs, Ys)), delimiter=",")
    ### Plotando o resultado principal - Espaço de fases
    pl.figure(figsize=(8,4), dpi=480)
    pl.plot(Xm, Ym, 'b-.', label="motor")
    pl.plot(Xs, Ys, 'r-', label="sac-rl")
    #pl.title("Paths to $x_0$ = "+str(list(x0)))
    pl.xlabel("$X$")
    pl.ylabel("$Y$")
    pl.legend()
    pl.savefig("prog10/imgs/efases{:02d}.png".format(ko+1))
    pl.clf()
    ###### CORRELAÇÕES
    ### Matrizes
    Xmin = np.amin([len(Xm), len(Xs)])
    Amin = np.amin([len(La1), len(teta1)])
    MCX = np.corrcoef(Xm[1:Xmin], Xs[1:Xmin]).real
    MCY = np.corrcoef(Ym[1:Xmin], Ys[1:Xmin]).real
    MCA1 = np.corrcoef(La1[1:Amin], teta1[1:Amin]).real
    MCA2 = np.corrcoef(La2[1:Amin], teta2[1:Amin]).real
    ### Plotando a matriz de correlação - X
    rotulos = ['Data','Forecast']
    pl.matshow(abs(MCX), cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
    pl.colorbar()
    pl.title('(Corr-Q1)', y=-0.1)
    pl.gca().set_xticklabels(['']+rotulos)
    pl.gca().set_yticklabels(['']+rotulos)
    pl.savefig("prog10/imgs/corr_q1{:02d}.png".format(ko+1), dpi=500)
    pl.clf()
    ### Plotando a matriz de correlação - Y
    pl.matshow(abs(MCY), cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
    pl.colorbar()
    pl.title('(Corr-Q2)', y=-0.1)
    pl.gca().set_xticklabels(['']+rotulos)
    pl.gca().set_yticklabels(['']+rotulos)
    pl.savefig("prog10/imgs/corr_q2{:02d}.png".format(ko+1), dpi=500)
    pl.clf()
    ### Plotando a matriz de correlação - Ângulo 1
    pl.matshow(abs(MCA1), cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
    pl.colorbar()
    pl.title('(Corr-$Theta_1$)', y=-0.1)
    pl.gca().set_xticklabels(['']+rotulos)
    pl.gca().set_yticklabels(['']+rotulos)
    pl.savefig("prog10/imgs/corr_a1{:02d}.png".format(ko+1), dpi=500)
    pl.clf()
    ### Plotando a matriz de correlação - Ângulo 2
    pl.matshow(abs(MCA2), cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
    pl.colorbar()
    pl.title('(Corr-$Theta_2$)', y=-0.1)
    pl.gca().set_xticklabels(['']+rotulos)
    pl.gca().set_yticklabels(['']+rotulos)
    pl.savefig("prog10/imgs/corr_a2{:02d}.png".format(ko+1), dpi=500)
    pl.clf()
    ### Salvando as informações
    np.savetxt("prog10/data/angulos{:02d}.csv".format(ko+1), VAG, delimiter=",")
    LAG = []  # Zerando a lista
    ### Reiniciando o sistema
    pendulo.reset()      # Reinicia o sistema
    ko += 1
    if (ko > 29):         # Até 30: ko = 0, 1, .., 29
      break

###### Montagem 1
if (len(LMo) > 0):
  numI = LMo[0]
  img06 = imread("prog10/imgs/trajcaos{:02d}.png".format(numI))
else:
  numI = 1
  img06 = imread("prog10/imgs/efases{:02d}.png".format(numI))
### Importando as imagens
img01 = imread("prog10/imgs/corr_q1{:02d}.png".format(numI))
img02 = imread("prog10/imgs/corr_q2{:02d}.png".format(numI))
img03 = imread("prog10/imgs/corr_a1{:02d}.png".format(numI))
img04 = imread("prog10/imgs/corr_a2{:02d}.png".format(numI))
img05 = imread("prog10/imgs/efases{:02d}.png".format(numI))
### Plotando a montagem
pl.figure(figsize=(8,4), dpi=1000)
# 1
pl.subplot(2, 3, 1)
pl.axis('off')
pl.imshow(img01)
pl.title("(A)", y=-0.1)
# 2
pl.subplot(2, 3, 2)
pl.axis('off')
pl.imshow(img02)
pl.title("(B)", y=-0.1)
# 3
pl.subplot(2, 3, 3)
pl.axis('off')
pl.imshow(img03)
pl.title("(C)", y=-0.1)
# 4
pl.subplot(2, 3, 4)
pl.axis('off')
pl.imshow(img04)
pl.title("(D)", y=-0.1)
# 5
pl.subplot(2, 3, 5)
pl.axis('off')
pl.imshow(img05)
pl.title("(E)", y=-0.2)
# 6
pl.subplot(2, 3, 6)
pl.axis('off')
pl.imshow(img06)
pl.title("(F)", y=-0.2)
### Salvando
pl.savefig("prog10/montagem1.png")
pl.clf()

###### Montagem 2
numI = 7
### Importando as imagens
img01 = imread("prog10/imgs/corr_q1{:02d}.png".format(numI))
img02 = imread("prog10/imgs/corr_q2{:02d}.png".format(numI))
img03 = imread("prog10/imgs/corr_a1{:02d}.png".format(numI))
img04 = imread("prog10/imgs/corr_a2{:02d}.png".format(numI))
### Plotando a montagem
pl.figure(figsize=(4,4), dpi=800)
# 1
pl.subplot(2, 2, 1)
pl.axis('off')
pl.imshow(img01)
pl.title("(A)", y=-0.1)
# 2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(img02)
pl.title("(B)", y=-0.1)
# 3
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(img03)
pl.title("(C)", y=-0.1)
# 4
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(img04)
pl.title("(D)", y=-0.1)
### Salvando
pl.savefig("prog10/montagem2.png")
pl.clf()

###### Montagem 3
### Importando as imagens
img01 = imread("prog10/imgs/corr_q1{:02d}.png".format(1))
img02 = imread("prog10/imgs/corr_q1{:02d}.png".format(2))
img03 = imread("prog10/imgs/corr_q1{:02d}.png".format(4))
img04 = imread("prog10/imgs/corr_q1{:02d}.png".format(7))
img05 = imread("prog10/imgs/corr_q1{:02d}.png".format(8))
img06 = imread("prog10/imgs/corr_q1{:02d}.png".format(9))
### Plotando a montagem
pl.figure(figsize=(8,4), dpi=1000)
# 1
pl.subplot(2, 3, 1)
pl.axis('off')
pl.imshow(img01)
pl.title("(A)", y=-0.1)
# 2
pl.subplot(2, 3, 2)
pl.axis('off')
pl.imshow(img02)
pl.title("(B)", y=-0.1)
# 3
pl.subplot(2, 3, 3)
pl.axis('off')
pl.imshow(img03)
pl.title("(C)", y=-0.1)
# 4
pl.subplot(2, 3, 4)
pl.axis('off')
pl.imshow(img04)
pl.title("(D)", y=-0.1)
# 5
pl.subplot(2, 3, 5)
pl.axis('off')
pl.imshow(img05)
pl.title("(E)", y=-0.1)
# 6
pl.subplot(2, 3, 6)
pl.axis('off')
pl.imshow(img06)
pl.title("(F)", y=-0.1)
### Salvando
pl.savefig("prog10/montagem3.png")
pl.clf()

### Salvando os alvos
np.savetxt("prog10/data/alvos.csv", LXY, delimiter=",")

### FIM """
### Referências
### [*] cos1: cosseno do ângulo da primeira junta
###     cos2: cosseno do ângulo da segunda junta
###     sin2: seno do ângulo da segunda junta
###     xo: coordenada X do objeto
###     vx_b: velocidade X do braço: dxb/dt
###     dx_bo: distância em X do braço ao objeto = |xb - xo|
###
### [1] https://stable-baselines3.readthedocs.io/en/master/
