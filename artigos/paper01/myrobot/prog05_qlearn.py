### Programa que o robô vai usar uma q-table para aprender a pegar três objetos
### Use seed

### As 9 Ações
A0 -> -1, -1     x dx = r
A1 -> -1, +1     x dx
A2 -> -1,  0     x dx
A3 -> +1, -1     x dx
A4 -> +1, +1     x dx
A5 -> +1,  0     x dx
A6 ->  0, -1     x dx
A7 ->  0, +1     x dx
A8 ->  0,  0  (Pegou o objeto) 

### Estados
=> Não use formas circulares:
Área Útil: AU = 0.5*pi*R²
Tamanho do objeto: To = pi*r²
Logo, se:
  R = 2
  r = 0.04   como normalmente usamos, progs 1 e 2, além de mylibs 2 e 3
Então temos:
  N° de estados = AU/To ~ 6.283/0.005 = 1250 estados
MAS NÃO É ISSO

Área Factível = Uma mesa de 2.8 m por 1.4
Área de cada Estado = Um objeto quadrado de lado 0.04
AF = 2.8*1.4 = 3.92 m²
AE = 0.04*0.04 = 0.0016
Então, AF/AE = 2450, este é o número de estados
Nº de Estados com recompensa = 2450

  Cada estado é definido pelas coordenadas: x e y
  Estado 0 ->   [ teta1=0, teta2=0]
  Estado 1 ->   Estado 0 + A0 = [-0.04, -0.04]
  Estado 2 ->   Estado 0 + A1 = [-0.04, +0.04]
  Estado 3 ->   Estado 0 + A2 = [-0.04,  0.00]
  Estado 4 ->   [+0.04, -0.04]
  ...
  Estado 9 ->   Estado 1 + A0 = [-0.08, -0.08]


q_table = np.zeros(2450, 9)

qual o valor de dteta? para reprsentar a ação ...?
