#!/bin/bash
### ShellScript para realizar 4 simulações
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Parâmetros iniciais 
K=04

### Executando o código
python3 lorenz_system.py

### Criando o vídeo
cd frames
ffmpeg -framerate 25 -i %03d.png movie_${K}.avi     ## Comando principal
rm *.png
cd ..
mv frames/* movies/

### Pronto
clear && tree

## FIM
