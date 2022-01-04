#!/bin/bash
### ShellScript para realizar 4 simulações
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Parâmetros iniciais 
K=02

### Executando o código
python3 vanderpol_system.py

### Criando o vídeo da posicao
cd posicao
ffmpeg -framerate 25 -i %03d.png posicao_${K}.avi     ## Comando principal
rm *.png
cd ..
mv posicao/* movies/

### Criando o vídeo das fases
cd efases
ffmpeg -framerate 25 -i %03d.png fases_${K}.avi     ## Comando principal
rm *.png
cd ..
mv efases/* movies/

### Pronto
clear && tree

## FIM
