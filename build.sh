CC_FLAGS="-I/usr/local/cuda/include -std=c99 -O3 -Wno-missing-braces -fopenmp -Wall"
LD_FLAGS="-lcudart -L/usr/local/cuda/lib64"

# Build MPI/OpenMP files
mpicc -c -o build/main.o src/main.c $CC_FLAGS
mpicc -c -o build/io.o src/io.c $CC_FLAGS

# Build CUDA files
nvcc -c src/update_tile.cu -o build/update_tile.o
nvcc -dlink build/update_tile.o -o build/update_tile_link.o 

# Link everything together
mpicc -o build/game_of_life build/io.o build/main.o build/update_tile_link.o build/update_tile.o $CC_FLAGS $LD_FLAGS
