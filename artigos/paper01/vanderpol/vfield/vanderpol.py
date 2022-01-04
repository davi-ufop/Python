### Plotando o campo vetorial do oscilador de Van der Pol
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando o nosso módulo
import matplotlib.pyplot as pl
from plot_vetor import *     ## plot_campo()

### Sistema dinâmico do oscilador de Duffing
vanderpol = ["Y", "8.53*(1 - X*X)*Y - X"]

### Plotando
pl.figure()
plot_campo(vanderpol, xran=[-2, 2], yran=[-2, 2])
pl.show()
