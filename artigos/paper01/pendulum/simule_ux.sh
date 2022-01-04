#!/bin/bash
### ShellScript para realizar 4 simulações
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Parâmetros iniciais 
K=01

### Executando o código
python3 double_pendulum.py

### Criando o vídeo da posicao
cd trajetorias
ffmpeg -framerate 25 -i %03d.png traj_${K}.avi     ## Comando principal
rm *.png
cd ..
mv trajetorias/* movies/

### Criando o vídeo das fases
cd estados
ffmpeg -framerate 25 -i %03d.png fase_${K}.avi     ## Comando principal
rm *.png
cd ..
mv estados/* movies/

### Pronto
clear && tree -I 'sim*'

## FIM
