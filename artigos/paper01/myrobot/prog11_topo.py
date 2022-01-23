### Programa para aferir parãmetros topológicos das trajetórias caóticas
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e modulos
from mylib10 import *
import nolds as nd

###### Locais das trajetórias:
### Prog06: Dados e Previsões
traj6_D = np.genfromtxt("prog06/data/out_0481.csv", delimiter=",")
traj6_P = np.genfromtxt("prog06/data/prev_0481.csv", delimiter=",")
### Prog07: Dados, Previsões e Simulações
traj7_D = np.genfromtxt("prog07/data/dpendulo/saida48.csv", delimiter=",")
traj7_P = np.genfromtxt("prog07/data/dpendulo/prev_48.csv", delimiter=",")
traj7_S = np.genfromtxt("prog07/data/dpendulo/sims_48.csv", delimiter=",")
### Prog10: Dados e Previsões
trajX_D = np.genfromtxt("prog10/data/motor_04.csv", delimiter=",")
trajX_P = np.genfromtxt("prog10/data/prevs_04.csv", delimiter=",")

"""
###### Dando uma olhada
### Dados 06
print("\nCNN -> Autoencoder")
print("D6: ", traj6_D[1:2])
print("P6: ", traj6_P[1:2])
### Dados 07
print("\nDynamic System - Koopman Method")
print("D7: ", traj7_D[1:2])
print("P7: ", traj7_P[1:2])
print("P7: ", traj7_S[1:2])
### Dados 10
print("\nQ-Learn - SAC")
print("DX: ", trajX_D[1:2])
print("PX: ", trajX_P[1:2])
"""

##### Tranforme figuras em curvas
YD = fig_to_curve(traj6_D, "prog11/imgs/traj6d.png")
YP = fig_to_curve(traj6_P, "prog11/imgs/traj6p.png")

###### MEDIDAS TOPOLÓGICAS DE 06
### Lyapunov, entropia, correlação e Hurst
lyp6D = nd.lyap_r(YD)
entrop6D = nd.sampen(YD)
#corr6D = nd.corr_dim(YD)
hurst6D = nd.hurst_rs(YD)

### Resultados
print("\nLiapunov 6D:", lyp6D)
print("\nEntropia 6D:", entrop6D)
#print("\nDim. Fractal 6D:", corr6D)
print("\nHurst 6D:", hurst6D)

### FIM
print("\nFIM")
