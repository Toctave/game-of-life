#pragma once

#include "geometry.h"

typedef struct {
    int width;
    int height;
    int tile_hcount;
    int tile_vcount;
    int tile_size;
    int wide_size; // == tile_size + 2 * margin_width
    int margin_width;
    
    Rect subregions_send[8];
    Rect subregions_recv[8];
    int subregion_sizes[8];

    int node_count;
} GridDimensions;

