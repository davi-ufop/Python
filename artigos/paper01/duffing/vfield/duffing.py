### Plotando o campo vetorial do oscilador de Duffing
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando o nosso módulo
import matplotlib.pyplot as pl
from plot_vetor import *     ## plot_campo()

### Sistema dinâmico do oscilador de Duffing
duffing = ["Y", "X - X**3"]

### Plotando
pl.figure()
plot_campo(duffing, xran=[-2, 2], yran=[-3, 3])
pl.show()
