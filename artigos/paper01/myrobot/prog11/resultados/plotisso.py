### Plot os resultados - Davi Neves - Ouro Preto, Brasil - Jan. 2022
# Bibliotecas
import numpy as np
import pylab as pl

# Dados -> 7
x = [1, 2, 3, 4]
lyap7D = [0.03, 0.01, 0.01, 0.02, 0.03]
lyap7P = [0.01, 0.01, 0.001, -0.001, 0.004]
hurst7D = [0.917, 0.919, 0.939, 0.918, 0.933, 0.927, 0.907, 0.923, 0.942, 0.931]
hurst7P = [0.943, 0.926, 0.960, 0.934, 0.950, 0.939, 0.935, 0.927, 0.919, 0.953]
shannon7D = [0.108, 0.096, 0.232, 0.098, 0.143, 0.102, 0.173]#, 0.123, 0.140, 0.116]
shannon7P = [0.125, 0.108, 0.252, 0.109, 0.206, 0.130, 0.216]#, 0.037, 0.020, 0.068]
pearson7 = [0.97, 0.96, 0.95, 0.91, 0.97]
lyapuno7 = np.array(lyap7D)/np.array(lyap7P)
hurstco7 = np.array(hurst7P)/np.array(hurst7D)
shannon7 = np.array(shannon7P)/np.array(shannon7D)

# Média
mp7 = np.mean(pearson7)/1
ml7 = np.mean(lyapuno7)/0.35
mh7 = np.mean(hurstco7)/max(hurstco7)
ms7 = np.mean(shannon7)/max(shannon7)
y = [mp7, ml7, mh7, ms7]

# Desvio -> erro
sp7 = np.std(pearson7)
sl7 = np.std(lyapuno7)/100
sh7 = np.std(hurstco7)
ss7 = np.std(shannon7)
e = [sp7, sl7, sh7, ss7]

# Plotando
labels = ['Pearson', 'Lyapunov', 'Hurst', 'Shannon']
pl.figure(figsize=(4,5.5), dpi=200)
pl.errorbar(x, y, e, linestyle='None', marker='D', mfc='red', mec='black', ms=7, mew=1.2, elinewidth=3)
pl.xticks(x, labels, rotation ='23')
pl.ylim(0.05, 1.05)
pl.savefig("resultado_koopman.png")

######################################################################################################
# Dados -> 6
lyap6D = [0.004, 0.004, 0.007, 0.005, 0.005]
lyap6P = [0.003, 0.001, 0.002, 0.003, 0.004]
hurst6D = [0.910, 0.995, 0.986, 1.017, 0.995, 0.693, 0.912, 0.768, 0.711, 1.088]
hurst6P = [0.792, 0.812, 0.755, 0.841, 0.812, 0.565, 0.690, 0.481, 0.495, 0.855]
shannon6D = [0.290, 0.418, 0.142, 0.137, 0.323, 0.203, 0.180]
shannon6P = [0.458, 0.973, 0.295, 0.294, 0.811, 0.348, 0.659]
pearson6 = [0.96, 0.96, 0.95, 0.95, 0.96]
lyapuno6 = np.array(lyap6D)/np.array(lyap6P)
hurstco6 = np.array(hurst6P)/np.array(hurst6D)
shannon6 = np.array(shannon6D)/np.array(shannon6P)

# Média
mp6 = np.mean(pearson6)/1
ml6 = np.mean(lyapuno6)/max(lyapuno6)
mh6 = np.mean(hurstco6)/max(hurstco6)
ms6 = np.mean(shannon6)/max(shannon6)
y = [mp6, ml6, mh6, ms6]

# Desvio -> erro
sp6 = np.std(pearson6)
sl6 = np.std(lyapuno6)/8
sh6 = np.std(hurstco6)
ss6 = np.std(shannon6)
e = [sp6, sl6, sh6, ss6]

# Plotando
labels = ['Pearson', 'Lyapunov', 'Hurst', 'Shannon']
pl.figure(figsize=(4,5.5), dpi=200)
pl.errorbar(x, y, e, linestyle='None', marker='D', mfc='red', mec='black', ms=7, mew=1.2, elinewidth=3)
pl.xticks(x, labels, rotation ='23')
pl.ylim(0.05, 1.05)
pl.savefig("resultado_decoder.png")

######################################################################################################
# Dados -> 10
lyapXD = [0.031, 0.045, 0.040, 0.040, 0.045]
lyapXP = [0.053, 0.057, 0.064, 0.056, 0.057]
hurstXD = [0.852, 0.860, 0.865, 0.865, 0.860, 0.866]
hurstXP = [0.857, 0.881, 0.822, 0.848, 0.881, 0.845]
shannonXD = [0.170, 0.206, 0.211, 0.210, 0.206, 0.139]
shannonXP = [0.084, 0.144, 0.045, 0.073, 0.144, 0.082]
pearsonX = [0.91, 0.93, 0.88, 0.91, 0.92, 0.92]
lyapunoX = np.array(lyapXD)/np.array(lyapXP)
hurstcoX = np.array(hurstXP)/np.array(hurstXD)
shannonX = np.array(shannonXP)/np.array(shannonXD)

# Média
mpX = np.mean(pearsonX)/1
mlX = np.mean(lyapunoX)/max(lyapunoX)
mhX = np.mean(hurstcoX)/max(hurstcoX)
msX = np.mean(shannonX)/max(shannonX)
y = [mpX, mlX, mhX, msX]

# Desvio -> erro
spX = np.std(pearsonX)
slX = np.std(lyapunoX)/8
shX = np.std(hurstcoX)
ssX = np.std(shannonX)
e = [spX, slX, shX, ssX]

# Plotando
labels = ['Pearson', 'Lyapunov', 'Hurst', 'Shannon']
pl.figure(figsize=(4,5.5), dpi=200)
pl.errorbar(x, y, e, linestyle='None', marker='D', mfc='red', mec='black', ms=7, mew=1.2, elinewidth=3)
pl.xticks(x, labels, rotation ='23')
pl.ylim(0.05, 1.05)
pl.savefig("resultado_qlearn.png")

### FIM
