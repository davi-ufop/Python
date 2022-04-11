# biblioteca de python
import numpy as np
# matriz randômica K, contendo números de 0 até 12
K = np.random.randint(12, size=(4,4))
# determinando as matrizes de decomposição
l, X = np.linalg.eig(K)   # autovalores e autovetores
L = np.diag(l)            # matriz de autovalores 
Liv = np.linalg.inv(L)    # inversa dos autovalores
Xiv = np.linalg.inv(X)    # inversa dos autovetores
# decomposição da matriz K e de sua inversa
Kp = X@L@Xiv
Kiv = X@Liv@Xiv
# confirmando que os cálculos estão corretos
Id = K@Kiv.real           # este produto deve resultar
print(np.round(Id,0))     # na matriz identidade: Ok!
