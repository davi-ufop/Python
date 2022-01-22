### Programa que utiliza o operador de Koopman para reconstruir
### as trajetórias no espaço de fases 
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos
import numpy as np          ### Numérica
import pylab as pl          ### Gráfica
import pykoopman as kp      ### Método de Koopman
from pykoopman.observables._custom_observables import CustomObservables
from tqdm import tqdm       ### Barras de progresso
from imageio import imread  ### Imagens
### Pra evitar warnings irrelevantes
import warnings
warnings.filterwarnings('ignore')

### Diretórios
DDP = "prog09/data/"
DIP = "prog09/imgs/"
DKP = "prog09/koopman/"

### Parâmetros de mylib06,py
N = 2 #50
t = np.arange(0, 20, 0.01)
dt = abs(t[0] - t[1])
Q = 100

###### Função para diferenciação
def diff(x, t):
  return np.gradient(x, axis=0)   ### Vide [1 e 2]

###### DADOS E AJUSTES
### Condições iniciais
x0 = np.genfromtxt(DDP+"entrada{:02d}.csv".format(N), delimiter=",")
### Trajetórias -> Dados: Data-Drive
X = np.genfromtxt(DDP+"saida{:02d}.csv".format(N), delimiter=",")
X1, X2, X3, X4 = X[:,0], X[:,1], X[:,2], X[:,3]
### Adequando os dados para usarmos no modelo -> Vide [1]
F = np.array([X1, X2, X3, X4]).T  ### Field 
###### OPERADOR DE KOOPMAN
### Determinando o operador (matriz) de Koopman
modelo1 = kp.Koopman()             ### Modelo Simples
modelo2 = kp.KoopmanContinuous(differentiator=diff)  ### Modelo Contínuo
### Fit dos dois modelos
modelo1.fit(F)                     ### Ajustando os dados
modelo2.fit(F, t=t)                ### Ajustando os dados
### Os dois operadores
KY = modelo1.koopman_matrix.real    ### Determinando a matrix do operador
KZ = modelo2.koopman_matrix.real    ### Determinando a matrix do operador
### Evolução dos estados
Y1, Z1 = x0, x0    ### Condição inicial
LY, LZ = [], []    ### Listas para registros
for i in range(Q):
  ### Registrando os estados
  LY.append(Y1)
  LZ.append(Z1)
  ### Evoluindo os dados - Método clássico/Série de Taylor
  Y1 = Y1 + (dt*KY).dot(Y1)
  Z1 = Z1 + (dt*KZ).dot(Z1)
  #Y1 = KY.dot(Y1)  ### Como definido nos artigos, mas 
  #Z1 = KZ.dot(Z1)  ### deveríamos conhecer a função g(X) - medida
### Adequando os resultados
Y, Z = np.array(LY), np.array(LZ)
Y1, Z1 = Y[:,0], Z[:, 0]
### Resultados
pl.plot(X1[0:Q], 'k-.', label='exata')
pl.plot(Y1, 'm-.', label='simples')
pl.plot(Z1, 'g-.', label='continuo')
pl.title("x0 = "+str(np.round(x0,3)))
pl.legend()
pl.savefig(DIP+"compare_modelos.png")
pl.clf()
### Mostre os operadores
print("\nK simples:\n", KY)
print("\nK continuo:\n", KZ)
print("\nDiferenças irrelevantes!")
### Salvando
np.savetxt(DKP+"koopman_simples.csv", KY, delimiter=",")
np.savetxt(DKP+"koopman_continuo.csv", KZ, delimiter=",")

### Outra observação importante - USANDO O MODELO SIMPLES!
Xp = modelo1.predict(F).real
Xs = modelo1.simulate(F[0], n_steps=Q)
### Resultado comparativo importante!
pl.plot(X1[1:Q], 'k-', label='dados') 
pl.plot(Xp[1:Q,0], 'r-', label='previsto') 
pl.plot(Xs[:,0], 'b-', label='simulado') 
pl.title("x0 = "+str(np.round(x0,3)))
pl.legend()
pl.savefig(DIP+"compare_modulos.png")
pl.clf()

### Agora as quatro curvas que foram extra-dados
pl.figure(figsize=(8, 4), dpi=128)
### Métodos clássicos
pl.subplot(1,2,1)
pl.plot(Y1, 'm-.', label='simples')
pl.plot(Z1, 'g-.', label='continuo')
pl.title("Clássicos")
pl.gca().axes.get_yaxis().set_visible(False)
pl.legend()
### Métodos de Kaiser/Silva
pl.subplot(1,2,2)
pl.plot(Xp[1:Q,0], 'r-', label='previsto') 
pl.plot(Xs[:,0], 'b-', label='simulado') 
pl.title("Kaiser/Silva")
pl.gca().axes.get_yaxis().set_visible(False)
pl.legend()
### Save
pl.savefig(DIP+"compare_metodos.png")
pl.clf()

###### TRABALHANDO COM OBSERVÁVEIS
### Construindo as medidas -> Espaçod as medidas
medidas = CustomObservables(F)
print("\nMedidas:", medidas)
### Construindo o modelo para o espaçod as medidas
modelo3 = kp.Koopman(observables=medidas)
### DESCUBRA COMO USAR! VIDE [3]

### FIM """ 
### [1] https://pykoopman.readthedocs.io/en/latest/
### [2] https://pykoopman.readthedocs.io/en/latest/examples/differentiation.html
### [3] https://pykoopman.readthedocs.io/en/latest/examples/observables.html
