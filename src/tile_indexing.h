#pragma once

#include "grid_dimensions.h"

static inline int index_of_tile(int tx, int ty, const GridDimensions* gd) {
    return ty * gd->tile_hcount + tx;
}

static inline void tile_of_index(int idx, int* tx, int* ty, const GridDimensions* gd) {
    *tx = idx % gd->tile_hcount;
    *ty = idx / gd->tile_hcount;
}

static inline int rank_of_index(int idx, const GridDimensions* gd) {
    return 1 + idx % (gd->node_count - 1); 
}

static inline int tile_count_of_rank(int rank, const GridDimensions* gd) {
    int worker_count = gd->node_count - 1;
    int total_count = gd->tile_hcount * gd->tile_vcount;
    
    int bonus = (rank - 1) < (total_count % worker_count);
    return total_count / worker_count + bonus;
}

static inline int rank_of_tile(int tx, int ty, const GridDimensions* gd) {
    int idx = index_of_tile(tx, ty, gd);
    return rank_of_index(idx, gd);
}

