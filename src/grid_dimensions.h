#pragma once

typedef struct {
    int width;
    int height;
    int tile_hcount;
    int tile_vcount;
    int tile_size;
    int wide_size; // == tile_size + 2

    int node_count;
} GridDimensions;

