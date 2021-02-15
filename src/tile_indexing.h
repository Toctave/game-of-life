#pragma once

#include "grid_dimensions.h"
#include "geometry.h"

typedef enum {
    TOPLEFT, TOP, TOPRIGHT,
    LEFT, RIGHT,
    BOTLEFT, BOT, BOTRIGHT,
    NEIGHBOUR_INDEX_COUNT,
} NeighbourIndex;

static inline int index_of_tile(Vec2i pos, const GridDimensions* gd) {
    return pos.y * gd->tile_hcount + pos.x;
}

static inline Vec2i tile_of_index(int idx, const GridDimensions* gd) {
    return (Vec2i) {
        idx % gd->tile_hcount, idx / gd->tile_hcount
    };
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

static inline int rank_of_tile(Vec2i pos, const GridDimensions* gd) {
    int idx = index_of_tile(pos, gd);
    return rank_of_index(idx, gd);
}

static inline Vec2i neighbour_tile(Vec2i tile, NeighbourIndex idx, const GridDimensions* gd) {
    Vec2i co = tile;
    
    switch (idx) {
    case TOPLEFT:
        co.y--;
        co.x--;
        break;
    case TOP:
        co.y--;
        break;
    case TOPRIGHT:
        co.y--;
        co.x++;
        break;
    case LEFT:
        co.x--;
        break;
    case RIGHT:
        co.x++;
        break;
    case BOTLEFT:
        co.y++;
        co.x--;
        break;
    case BOT:
        co.y++;
        break;
    case BOTRIGHT:
        co.y++;
        co.x++;
        break;
    default:
        return tile;
    }

    co.x = (co.x + gd->tile_hcount) % gd->tile_hcount;
    co.y = (co.y + gd->tile_vcount) % gd->tile_vcount;

    return co;
}
