"""
Programa para resolver o problema de Navier-Stokes do fluxo externo em
torno de cracas no caso de um navio, numa escala bidimensional.

Base deste programa: navier03.py

Elaborado por Davi Neves - UFOP/Brasil - Maio de 2022
"""

### Bibliotecas necessárias
import tqdm as td         # Barra de progresso - dispensável
import numpy as np        # Numérica
import pylab as pl        # Gráfica
import mshr as msh        # MESH
import fenics as fcs      # FEniCS

### Parâmetros do problema
T = 0.1    # Tempo final
N = 50     # Número de passos
dt = T/N   # Passo temporal
mu = 0.002    # Viscosidade dinâmica [Pa.s] 
rho = 1       # Densidade [kg/m³]
## Adequação necessária
k  = fcs.Constant(dt)
mu = fcs.Constant(mu)
rho = fcs.Constant(rho)

### Criando o MESH do problema
casco = msh.Rectangle(fcs.Point(0, 0), fcs.Point(3.0, 1.5)) 
craca1 = msh.Circle(fcs.Point(0.2, 0), 0.10)                          
craca2 = msh.Circle(fcs.Point(0.6, 0), 0.09)                          
craca3 = msh.Circle(fcs.Point(1.0, 0), 0.11)                          
craca4 = msh.Circle(fcs.Point(1.6, 0), 0.10)                          
craca5 = msh.Circle(fcs.Point(1.9, 0), 0.08)                          
craca6 = msh.Circle(fcs.Point(2.3, 0), 0.10)                          
craca7 = msh.Circle(fcs.Point(2.6, 0), 0.09)        
cracas = (craca1 + craca2 + craca3 + craca4 + craca5 + craca6 + craca7)                  
omega = casco - cracas                  # Região onde descreveremos o problema
mesh = msh.generate_mesh(omega, 64)     # MESH - Rede de elementos
### Apresentando o MESH
fcs.plot(mesh)
pl.show()
pl.cla()

### Definindo os espaços de funções: pressão e velocidade
V = fcs.VectorFunctionSpace(mesh, 'P', 2)   # Velocidade (vetorial)
Q = fcs.FunctionSpace(mesh, 'P', 1)         # Pressão (escalar)
T = fcs.TensorFunctionSpace(mesh, 'P', 2)   # Tensor de tensões

### Definindo as fronteiras do problema - STRINGS
inflow = 'near(x[0], 0)'                        # Entrada do fluido
outflow = 'near(x[0], 3.0)'                     # Saída do fluido
walls = 'near(x[1], 1.5)'                       # Parede superior
# Região das cracas, com refinamento do mesh [FEM]
corpos = 'on_boundary'

### Definindo as condições de contorno
# Condições de contorno da velocidade
bcu_inflow = fcs.DirichletBC(V, fcs.Constant((0.3, 0)), inflow)     # V_in = 30 cm/s
bcu_walls = fcs.DirichletBC(V, fcs.Constant((0.3, 0)), walls)       # V_wall = 30 cm/s
bcu_corpos = fcs.DirichletBC(V, fcs.Constant((0, 0)), corpos)       # V_body = 0 m/s
bcu = [bcu_inflow, bcu_walls, bcu_corpos]           
# Condições de contorno da pressão
bcp_outflow = fcs.DirichletBC(Q, fcs.Constant(0), outflow)          # P_out = 0 Pa
bcp = [bcp_outflow]

### Definindo as funções tentativas e testes
# Velocidade
u = fcs.TrialFunction(V)
v = fcs.TestFunction(V)
# Pressão
p = fcs.TrialFunction(Q)
q = fcs.TestFunction(Q)

### Definindo funções para armazenar as soluções
uN = fcs.Function(V)
uT = fcs.Function(V)
pN = fcs.Function(Q)
pT = fcs.Function(Q)

### Definindo expressões usadas no problema variacional
U  = 0.5*(uN + u)
n  = fcs.FacetNormal(mesh)
f  = fcs.Expression(('5*cos((1.57*x[1])/1.5)', '25*cos(2*(x[0]/0.0955)) + 25'), degree=2)

### Definindo o gradiente simétrico -> Mec.Flu.
def epsilon(u):
    return fcs.sym(fcs.nabla_grad(u))

### Definindo o tensor de tensões -> Mec.Con.
def sigma(u, p):
    return 2*mu*epsilon(u) - p*fcs.Identity(len(u))

### Tensão
def stress_fun(u, p):
  stress_visc = 2*fcs.sym(fcs.grad(u))
  stress = -p*fcs.Identity(2) + stress_visc
  return stress

### Definindo o cerne do problema variacional no passo 1
F1 = rho*fcs.dot((u - uN)/k, v)*fcs.dx + rho*fcs.dot(fcs.dot(uN, fcs.nabla_grad(uN)), v)*fcs.dx + fcs.inner(sigma(U, pN), epsilon(v))*fcs.dx + fcs.dot(pN*n, v)*fcs.ds - fcs.dot(mu*fcs.nabla_grad(U)*n, v)*fcs.ds - fcs.dot(f, v)*fcs.dx
a1 = fcs.lhs(F1)
L1 = fcs.rhs(F1)

### Definindo o cerne do problema variacional no passo 2
a2 = fcs.dot(fcs.nabla_grad(p), fcs.nabla_grad(q))*fcs.dx
L2 = fcs.dot(fcs.nabla_grad(pN), fcs.nabla_grad(q))*fcs.dx - (1/k)*fcs.div(uT)*q*fcs.dx

### Definindo o cerne do problema variacional no passo 3
a3 = fcs.dot(u, v)*fcs.dx
L3 = fcs.dot(uT, v)*fcs.dx - k*fcs.dot(fcs.nabla_grad(pT - pN), v)*fcs.dx

### Matrizes dos Assembles
A1 = fcs.assemble(a1)
A2 = fcs.assemble(a2)
A3 = fcs.assemble(a3)

### Aplicando as condições de contorno às matrizes
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

### Integração temporal
t = 0  # Tempo inicial
vtkU = fcs.File("sol02/u/u.pvd")
vtkP = fcs.File("sol02/p/p.pvd")
vtkS = fcs.File("sol02/s/s.pvd")
for n in td.tqdm(range(N)):
    ### Atualizando o passo temporal
    t += dt
    ### Passo 1: Tentativa da velocidade
    b1 = fcs.assemble(L1)
    [bc.apply(b1) for bc in bcu]
    fcs.solve(A1, uT.vector(), b1, 'bicgstab', 'hypre_amg')

    ### Passo 2: Correção da pressão
    b2 = fcs.assemble(L2)
    [bc.apply(b2) for bc in bcp]
    fcs.solve(A2, pT.vector(), b2, 'bicgstab', 'hypre_amg')

    ### Passo 3: Correção da velocidade
    b3 = fcs.assemble(L3)
    fcs.solve(A3, uT.vector(), b3, 'cg', 'sor')
    
    ### Atualizando os valores de uN e pN
    uN.assign(uT)
    pN.assign(pT)

    ### Salvando tensões pra calcular as forças DRAG E LIFT
    stressN = fcs.project(stress_fun(uT, pT), T)
    vtkS << stressN    

    ### Salvando as soluções 
    # Velocidade
    vtkU << uT
    # Pressão
    vtkP << pT

### Referência bibliográfica
## FEM - Solving PDEs in Python – The FEniCS Tutorial Volume I

### FIM
## Para executar este código: $ python3 navier03.py
## Pra visualizar os resultados use o ParaView
