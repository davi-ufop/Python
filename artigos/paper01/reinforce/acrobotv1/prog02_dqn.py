import numpy as np
import matplotlib.pyplot as pl
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

modelo = DQN('MlpPolicy', 'Acrobot-v1', verbose=1, exploration_final_eps=0.1, target_update_interval=250)

dpendulo = gym.make('Acrobot-v1')

### Recompensas
media, padrao = evaluate_policy(modelo, dpendulo, n_eval_episodes=10, deterministic=True)
print(media, padrao)


### [1] https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/
