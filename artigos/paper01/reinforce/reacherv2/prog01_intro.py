### Programa para manipular um braço robótico 2D, sem política de aprendizado, 
### ou seja, com comportamento randômico. Apenas pra introduzir o ambiente!
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando os módulos necessários
import numpy as np               ## Numérico                        
import matplotlib.pyplot as pl   ## Gráficos
import gym            ## Ambiente para RL
import mujoco_py
from random import uniform   ## Randômica
from time import sleep       ## Dá um tempo!

### Parâmetros da simulação
Passos = 200                ## Número de passos, use múltiplos de 100
tamanho = 0.025             ## Tamanho (raio) do objeto (m)
passo = 0                   ## Contadores 
sucesso = 0
ele_acha = 0
lim_a = 3.5                ## Limite para os valores das ações - ALTERE ISSO!

### Listas para plotagem
XP = []
YP = []
XT = []
YT = []
PT = []

### Ambiente do braço robótico 2D: Reacher-v2
pega = gym.make('Reacher-v2')
pega.reset()        ## Iniciando
pega.render()       ## Renderização

### Processo randômico do braço pegando objetos
for i in range(Passos):
  ### Contando passos executados
  passo += 1
  #print("Passo = ", passo)
  ### Ação escolhida por processo randômico -> altere estes limites!
  acao = np.array([uniform(-lim_a, lim_a), uniform(-lim_a, lim_a)])
  #print("Ação = ", acao)
  ### Passando para o próximo estado
  observacao, recompensa, condicao, info = pega.step(acao)
  ### -> observacao: cos1, cos2, sin1, sin2, x_target, y_target, vx_finger, vy_finger, distância(x,y,z)
  #print(observacao)
  ### Coletando os resultados do próximo estado:
  xpega = pega.get_body_com("fingertip")[0]   ## Posição do pegador
  ypega = pega.get_body_com("fingertip")[1]
  xalvo = pega.get_body_com("target")[0]      ## Posição do objeto = x_target acima
  yalvo = pega.get_body_com("target")[0]      ## = y_target
  ### Registrando estes mesmos resultados:
  XP.append(xpega)
  YP.append(ypega)
  XT.append(xalvo)
  YT.append(yalvo)
  PT.append(passo)
  ### Representação deste próximo estado:
  pega.render()
  ### Condição de sucesso:
  dx = abs(xpega-xalvo)  ## Distância entre pegador e seu alvo
  dy = abs(ypega-yalvo)
  if (dx < tamanho and dy < tamanho):
    ## Troca a posição do alvo
    pega.reset()
    ## Contabiliza o sucesso
    sucesso += 1
  ## Mostrando um equívoco do pacote:
  if (condicao == True):
    ## Contando o que ele acha que é sucesso!
    pega.reset()
    ele_acha += 1
  ## Dá um tempo, pra gente ver o que acontece!
  sleep(0.02)

### Fechando o ambiente de simulação
pega.close()

### Plotando os gráficos dos resultados
### Posição X
pl.plot(PT, XP, 'r-.')
pl.plot(PT, XT, 'b-.')
pl.title('Posições X do pegador(red) e alvo(blue)')
pl.xlabel('Passos')
pl.ylabel('Coordenada X')
pl.savefig('coordenadax.png')
pl.clf()
### Posição Y
pl.plot(PT, YP, 'r-.')
pl.plot(PT, YT, 'b-.')
pl.title('Posições Y do pegador(red) e alvo(blue)')
pl.xlabel('Passos')
pl.ylabel('Coordenada Y')
pl.savefig('coordenaday.png')
pl.clf()

### Resultado final:
print("\nDepois de ", Passos, " passos, o robô coletou ", sucesso, " alvos.")
print("O ambiente Reacher-v2 considerou que houve ", ele_acha, " sucessos.")
print("Acredito que foram considerados ", (Passos//50)-1, " sucessos. Acertei?")

### FIM
