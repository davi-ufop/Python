### Programa pra gerar os dados usados no Método DMD e 
### no programa do Operador de Koopman
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos
import numpy as np                  # Númerica
from scipy.integrate import odeint  # ODEs

### Parâmetros
alfa = 10 
beta = 8/3
gama = 23
x0 = [4.3, 1.8, 2.7]
t = np.arange(0, 50, 0.01)

### Sistema de Lorenz
def lorenz(x, t):
  x1, x2, x3 = x
  dx1_dt = alfa*(x2 - x1)
  dx2_dt = x1*(gama - x3) - x2
  dx3_dt = x1*x2 - beta*x3
  return dx1_dt, dx2_dt, dx3_dt

### Solução do sistema
X = odeint(lorenz, x0, t)

### Salvando em CSV
np.savetxt("lorenz.csv", X, delimiter=",")

### FIM
