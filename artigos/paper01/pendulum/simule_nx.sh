#!/bin/bash
### Programa para realizar N vezes as simulações
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Parâmetros
C=1
CR=$C
K0=3.0
#K0=3.5

for k in 3.1 3.2 3.3 3.4 3.5
#for k in 3.6 3.7 3.8 3.9 4.0
do
  ### Parâmetros
	C0=$C
  C=$(bc <<< $C"+ 1")
	K1=$(bc <<< $k"- 0.1")
  K2=$k
	### Realizando as simulações
  echo " " >> condicoes.txt
	echo "Simulação $C0:" >> condicoes.txt
  grep -i "CI = " double_pendulum.py >> condicoes.txt	
  bash simule_ux.sh
	### Alterando as CIs e o índice K dos resultados
  sed -i 's/CI = '$K1'/CI = '$K2'/g' double_pendulum.py
  sed -i 's/K=0'$C0'/K=0'$C'/g' simule_ux.sh
done

### Restaurando os arquivos
sed -i 's/CI = '$K2'/CI = '$K0'/g' double_pendulum.py
sed -i 's/K=0'$C'/K=0'$CR'/g' simule_ux.sh

### FIM
