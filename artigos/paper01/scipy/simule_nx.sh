#!/bin/bash
### ShellScript para realizar 4 simulações
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Parâmetros iniciais 
k0=7.0    ## Primeira condição inicial
dv=10     ## Variação das condições

### Loop para realizar as 4 simulações
for k in 0 1 2 3 4 5 6 7 8 9
do
  ## Arquivo pra registrar as condições
	echo " " >> condicoes.txt			
	echo "C.I. $k:" >> condicoes.txt
  grep -A 1 'iniciais' double_pendulum.py >> condicoes.txt
	## Realizando a simulação pra condição k
  python3 double_pendulum.py
	## Gerando o vídeo (AVI) com as soluções (PNG)
  mkdir "movie_${k}"                 ## Pasta para as figuras
  mv frames/*.png movie_${k}/        ## Movendo as figuras
	cd movie_${k}/                     ## Mudando pra pasta das figuras
  rename 's/_img0//' *.png           ## Renomeando elas adequadamente
	ffmpeg -framerate 25 -i %03d.png movie_${k}.avi     ## Comando principal
	cd ..                              ## Saindo e limpando a tela
	clear
	mv movie_${k}/*.avi movies/
  ## Trocando os valores das condições: k1 -> k2
	dk=$(echo "scale=1;" $k "/" $dv | bc -l)
  k1=$(bc <<< $k0"+"$dk)                       ## Variáveis de troca
  k2=$(bc <<< $k1"+ 0.1")
  sed -i 's/'$k1'/'$k2'/g' double_pendulum.py  ## Comando principal
	echo "Nova simulação!"
  grep -A 1 'iniciais' double_pendulum.py      ## Confirmando a troca
done   ## Pronto!
## Voltando a primeira condição
#sed -i 's/'$k2'/'$k0'/g' double_pendulum.py
## Limpando tudo
rm -rf movie_*
clear && ls ##&& tree
## FIM
