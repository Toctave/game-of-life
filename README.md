# Game of life

A C implementation of Conway's Game of Life.

## Dependencies :

- SDL2 for displaying the cells

## Building :

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Usage :

Set the width, height and initial population density from the command line :

```
./game_of_life 400 400 .5
```
