### Introdução ao AC-Robot do módulo GYM
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos
import numpy as np
import matplotlib.pyplot as pl
import gym
#import mujoco_py
from tqdm import tqdm

### Sistema
dpendulo = gym.make('Acrobot-v1')
estado = dpendulo.reset()

### Fazendo o pêndulo movimentar
R = 2
L1, L2 = R/2, R/2
lx, ly = [], []
for i in tqdm(range(5000)):
  acao = dpendulo.action_space.sample()
  proximo, recompensa, pronto, info = dpendulo.step(acao)
  #dpendulo.render()
  ### Ponta do pêndulo
  x = L1*proximo[1] + L2*proximo[3]
  y = -abs(L1*proximo[0] + L2*proximo[2])
  lx.append(x)
  ly.append(y)

pl.plot(lx, ly, 'r-.')
pl.show()

### FIM """
### [1] https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
