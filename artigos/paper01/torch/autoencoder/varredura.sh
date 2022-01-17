#!/bin/bash
for i in {10..480..10}
do
  echo "$i /480"
  #sed -i 's/k+'$i'/k+'$((i+10))'/' prog03_gtraj.py
	sed -i 's/k+'$i'/k+'$((i+10))'/' prog04_gtraj.py
  #python3 prog03_gtraj.py
  python3 prog04_gtraj.py
	#mv .trajetorias/* data/dpendulum/lowedef/imgs/
	mv .trajetorias/* data/dpendulum/highdef/imgs/
	clear
done
echo ' '
echo 'Voltando ao normal ... '
echo ' '
#sed -i 's/k+'$((i+10))'/k+10/' prog03_gtraj.py
sed -i 's/k+'$((i+10))'/k+10/' prog04_gtraj.py
#grep -i 'k+10' prog03_gtraj.py
grep -i 'k+10' prog04_gtraj.py
exit
