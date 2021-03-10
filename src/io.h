#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "grid_dimensions.h"

typedef struct {
    int node_count;
    int thread_count;
    int gpu_count;
    
    int width;
    int height;
    float density;
    bool gui_on;
    int iter;
    int seed;
    const char* input_filepath;
    const char* output_filepath;
    char log_filepath[256];
    
    int tile_size;
    int margin_width;
} Options;

bool parse_options(Options* options, int argc, char** argv);
void parse_rle_file(const char* filepath, uint8_t* tiles, const GridDimensions* gd);
void save_grid_to_png(const uint8_t* tiles, const char* filepath, const GridDimensions* gd);
void print_grid(const uint8_t* tiles, const GridDimensions* gd);

