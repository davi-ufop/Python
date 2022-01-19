### Para auxiliar a simulação realizada pelo programa 08
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Importando módulos necessários
from mylib04 import *
from random import seed, uniform
import warnings
warnings.filterwarnings('ignore')

### Parâmetros e variáveis do processo
R = 2             ### Dimensões do ambiente 
B = 1.2           ### Alcance máximo do braço
P = 2             ### Precisão das medidas
AN = 9            ### Número de ações
dd = 0.04         ### Precisão dos estados
K = 300000        ### Número de treinos
NumTab = 300

############ INICIANDO A CONTRUÇÃO DA Q-TABLE
### Criando estados discretos XY
saved = "prog08/tmp/estados.png"
EN, TAG, TXY = plot_estados(R, dd, saved)

### Verificando se todos os estados estão contemplados
Bom, Ruim = completo(R, dd, TXY)
print("\nDeu Ruim = ", Ruim)
print("Deu Bom = ", Bom)

### Criando a tabela Q zerada
qtab = np.zeros([EN, AN])
print("Número de estados = ", len(qtab))

### Loop para criar as tabelas Q
seed(888)
for i in tqdm(range(NumTab)):
  xo = round(dd*(uniform(-B, B)//dd), P)   ### Estado do objeto
  yo = round(dd*(uniform(-B, 0)//dd), P)
  estado = [round(xo, P), round(yo, P)]
  np.savetxt("prog08/data/state{:03d}.csv".format(i+1), estado, delimiter=',') ### CSV 
  IO = np.where(np.all(TXY == (xo, yo), axis=1))[0][0]  ### Índice do Objeto
  qtab[IO, 8] = 20    ### Recompensa por achar o objeto
  qtab = treino(qtab, K, TXY, xo, yo, dd, B)                                   ### Treino
  np.savetxt("prog08/data/qtable{:03d}.csv".format(i+1), qtab, delimiter=',')  ### CSV 
  ### Conferindo

### FIM
