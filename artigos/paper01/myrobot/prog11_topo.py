### Programa para aferir parãmetros topológicos das trajetórias caóticas
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e modulos
from mylib10 import *
import nolds as nd

###### IMPORTANDO OS DADOS
from random import choice
### Prog06: Dados e Previsões
N6 = 487#choice(np.arange(477, 499))
print("N6: ", N6)
traj6_D = np.genfromtxt("prog06/data/out_{:04d}.csv".format(N6), delimiter=",")
traj6_P = np.genfromtxt("prog06/data/prev_{:04d}.csv".format(N6), delimiter=",")
### Prog07: Dados, Previsões e Simulações
N7 = 19#choice(np.arange(10, 49))
print("N7: ", N7)
traj7_D = np.genfromtxt("prog07/data/dpendulo/saida{:02d}.csv".format(N7), delimiter=",")
traj7_P = np.genfromtxt("prog07/data/dpendulo/prev_{:02d}.csv".format(N7), delimiter=",")
traj7_S = np.genfromtxt("prog07/data/dpendulo/sims_{:02d}.csv".format(N7), delimiter=",")
### Prog10: Dados e Previsões
NX = 6#choice(np.arange(1, 9))
print("NX: ", NX)
trajX_D = np.genfromtxt("prog10/data/motor_{:02d}.csv".format(NX), delimiter=",")
trajX_P = np.genfromtxt("prog10/data/prevs_{:02d}.csv".format(NX), delimiter=",")

###### TRATAMENTO DOS DADOS
### PROG06 -> Tranformnado as figuras em curvas
YD = fig_to_curve1(traj6_D, "prog11/imgs/traj6d.png")   # 452
YP = fig_to_curve2(traj6_P, "prog11/imgs/traj6p.png")   # ???, varia com a figura
### PROG07 -> Pegando apenas a primeira coordenada
X1D = traj7_D[:,0]  # 2000
X1P = traj7_P[:,0]  # 2000
X1S = traj7_S[:,0]  # 700

###### MEDIDAS TOPOLÓGICAS DE 06
### Lyapunov, entropia, correlação e Hurst
lyp6D = nd.lyap_r(YD)
entrop6D = nd.sampen(YD)
#corr6D = nd.corr_dim(YD)
hurst6D = nd.hurst_rs(YD)
### Lyapunov, entropia, correlação e Hurst
#lyp6P = nd.lyap_r(YP)
entrop6P = nd.sampen(YP)
#corr6P = nd.corr_dim(YP)
hurst6P = nd.hurst_rs(YP)

###### MEDIDAS TOPOLÓGICAS DE 07
### Lyapunov, entropia e Hurst
lyp7D = nd.lyap_r(X1D)
entrop7D = nd.sampen(X1D)
hurst7D = nd.hurst_rs(X1D)
### Lyapunov, entropia e Hurst
lyp7P = nd.lyap_r(X1P)
entrop7P = nd.sampen(X1P)
hurst7P = nd.hurst_rs(X1P)
### Lyapunov, entropia e Hurst
lyp7S = nd.lyap_r(X1S)
entrop7S = nd.sampen(X1S)
hurst7S = nd.hurst_rs(X1S)

###### MEDIDAS TOPOLÓGICAS DE 10
### Lyapunov, entropia e Hurst
lypXD = nd.lyap_r(trajX_D)
entropXD = nd.sampen(trajX_D)
hurstXD = nd.hurst_rs(trajX_D)
### Lyapunov, entropia e Hurst
lypXP = nd.lyap_r(trajX_P)
entropXP = nd.sampen(trajX_P)
hurstXP = nd.hurst_rs(trajX_P)

###### RESULTADOS
### Prog06
# Lyapunov, entropia, correlação e Hurst
print("\nProg06:")
print("\tLiapunov 6D:", round(lyp6D, 3))
print("\tEntropia 6D:", round(entrop6D, 3))
#print("\tDim. Fractal 6D:", round(corr6D, 3))
print("\tHurst 6D:", round(hurst6D, 3))
# Entropia e Hurst
print("\n\tEntropia 6P:", round(entrop6P, 3))
print("\tHurst 6P:", round(hurst6P, 3))

### Prog07
# Lyapunov, entropia e Hurst
print("\n\nProg07:")
print("\tLiapunov 7D:", round(lyp7D, 3))
print("\tEntropia 7D:", round(entrop7D, 3))
print("\tHurst 7D:", round(hurst7D, 3))
# Lyapunov, entropia e Hurst
print("\n\tLiapunov 7P:", round(lyp7P, 3))
print("\tEntropia 7P:", round(entrop7P, 3))
print("\tHurst 7P:", round(hurst7P, 3))
# Lyapunov, entropia e Hurst
print("\n\tLiapunov 7S:", round(lyp7S, 3))
print("\tEntropia 7S:", round(entrop7S, 3))
print("\tHurst 7S:", round(hurst7S, 3))

### Prog10
# Lyapunov, entropia e Hurst
print("\n\nProg10:")
print("\tLiapunov XD:", round(lypXD, 3))
print("\tEntropia XD:", round(entropXD, 3))
print("\tHurst XD:", round(hurstXD, 3))
# Lyapunov, entropia e Hurst
print("\n\tLiapunov XP:", round(lypXP, 3))
print("\tEntropia XP:", round(entropXP, 3))
print("\tHurst XP:", round(hurstXP, 3))

### FIM
print("\nFIM")
