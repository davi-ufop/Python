# biblioteca de python
import numpy as np
# matriz randômica A, contendo números de 0 até 8
A = np.random.randint(8, size=(3,4))
print("A:\n", A)
# matriz dos valores singulares
S = np.zeros((3,4))  # menor termo remete às linhas
# determinando as matrizes de decomposição
U, s, VT = np.linalg.svd(A)  # autovalores e autovetores
V = VT.T  # transposta de VT == V
# preenchendo a matriz de valores singulares
S[:A.shape[0], :A.shape[0]] = np.diag(s) # se menor termo = linhas 
#S[:A.shape[1], :A.shape[1]] = np.diag(s) # se menor termo = colunas 
# inversa de S
Siv = np.linalg.pinv(S)
# decomposição da matriz A e de sua inversa
Ad = U@S@VT
print("A decomposta:\n", np.round(Ad.real,0))
Aiv = V@Siv@U.T
# confirmando que os cálculos estão corretos
Id = Ad.real@Aiv.real     # este produto deve resultar
print("I:\n",np.round(Id,0))     # na matriz identidade: Ok!
# fim
