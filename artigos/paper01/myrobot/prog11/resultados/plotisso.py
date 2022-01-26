### Plot os resultados - Davi Neves - Ouro Preto, Brasil - Jan. 2022
# Bibliotecas
import numpy as np
import pylab as pl

# Dados
x = [1, 2, 3, 4, 5]
pearson7 = [0.97, 0.96, 0.95, 0.91, 0.97]
hurst7D = [0.995, 1.044, 0.986, 1.024, 1.083]
hurst7P = [0.812, 0.873, 0.755, 0.732, 0.855]
shannon7D = [0.42,  0.315, 0.142, 0.175, 0.392]
shannon7P = [0.97,  0.873, 0.295, 0.847, 1.099]
shannon7D = np.array(shannon7D)/0.5
shannon7P = np.array(shannon7P)

# Média
mp7 = np.mean(pearson7)
mh7d = np.mean(hurst7D)
mh7p = np.mean(hurst7P)
ms7d = np.mean(shannon7D)
ms7p = np.mean(shannon7P)
y = [mp7, mh7d, mh7p, ms7d, ms7p]

# Desvio -> erro
sp7 = np.std(pearson7)
sh7d = np.std(hurst7D)
sh7p = np.std(hurst7P)
ss7d = np.std(shannon7D)
ss7p = np.std(shannon7P)
e = [sp7, sh7d, sh7p, ss7d, ss7p]

# Plotando
labels = ['Pearson', 'Hurst-1', 'Hurst-2', 'Shannon-1', 'Shannon-2']
pl.errorbar(x, y, e, linestyle='None', marker='s', mfc='red', mec='green', ms=15, mew=1.5, elinewidth=3.5)
pl.xticks(x, labels, rotation ='50')
pl.ylim(0.2, 1.2)
pl.show()

### FIM
