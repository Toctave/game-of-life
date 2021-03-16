read -p "Bon pour le CPU (marges < 4Ko & tiles petites) : Appuyer sur une touche pour continuer ..."
echo 
time salloc -n 10 -N 10 mpirun ./build/game_of_life -w 256 -h 256 -f patterns/gosper_glider_gun.rle --margin-width 4 -t 64 -o glider_gun_cpu.png -i 1000 --no-gpu
#time salloc -n 2 -N 2 mpirun ./build/game_of_life -w 256 -h 256 -f patterns/gosper_glider_gun.rle --margin-width 4 -t 64 -o glider_gun_cpu.png -i 1000 --no-gpu

read -p "Bon pour le GPU (tiles gigantesques) : Appuyer sur une touche pour continuer ..."
time salloc -n 10 -N 10 mpirun ./build/game_of_life -w 4096  -h 4096 -f patterns/lineship2.rle --margin-width 256 -t 1024 -i 10000 -o lineship2_gpu.png
#time salloc -n 2 -N 2 mpirun ./build/game_of_life -w 4096  -h 4096 -f patterns/lineship2.rle --margin-width 256 -t 2048 -i 10000 -o lineship2_gpu.png
