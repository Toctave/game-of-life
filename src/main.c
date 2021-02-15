#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include "io.h"
#include "tile_indexing.h"

typedef struct {
    int x;
    int y;
    uint8_t* cells[2];
} Tile;

typedef struct {
    int x;
    int y;
} Vec2i;

typedef struct {
    Vec2i min;
    Vec2i max;
} Rect;

static uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

typedef enum {
    TOPLEFT, TOP, TOPRIGHT,
    LEFT, RIGHT,
    BOTLEFT, BOT, BOTRIGHT,
    NEIGHBOUR_INDEX_COUNT,
} NeighbourIndex;

static void send_to_neighbour(uint8_t* buffer, size_t size, int dstx, int dsty, int my_index, const GridDimensions* gd) {
    int neighbour = rank_of_tile(dstx, dsty, gd);
    int tag = (my_index << 16) | index_of_tile(dstx, dsty, gd);
    MPI_Send(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD);
}

static void recv_from_neighbour(uint8_t* buffer, size_t size, int srcx, int srcy, int my_index, const GridDimensions* gd) {
    int neighbour = rank_of_tile(srcx, srcy, gd);
    int tag = (index_of_tile(srcx, srcy, gd) << 16) | my_index;

    // printf("receiving data from tile (%d, %d)\n", tag, srcx, srcy);
    MPI_Recv(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD, NULL);
}

static void update_tile_inside(uint8_t* src, uint8_t* dst, int wide_size, int margin_width) {
    int neighbour_offsets[8] = {
        -wide_size - 1, -wide_size, -wide_size + 1,
        -1,                                      1,
        wide_size - 1,   wide_size,  wide_size + 1
    };

    int start = margin_width;
    int end = wide_size - margin_width;

    /* Update internal tile */
    for (int j = start; j < end; j++) {
        for (int i = start; i < end; i++) {
            int neighbours = 0;
            int base = j * wide_size + i;
            for (int k = 0; k < 8; k++) {
                neighbours += src[base + neighbour_offsets[k]];
            }
            dst[base] = rule(src[base], neighbours);
        }
    }  
}

static void pack_sub_grid(const uint8_t* cells, int wide_size, Rect rect, uint8_t* buffer) {
    int row_length = (rect.max.x - rect.min.x);
    for (int y = rect.min.y; y < rect.max.y; y++) {
        int cells_offset = y * wide_size + rect.min.x;
        int buffer_offset = (y - rect.min.y) * row_length;
        
        memcpy(buffer + buffer_offset, cells + cells_offset, row_length);
    }
}

static void unpack_sub_grid(uint8_t* cells, int wide_size, Rect rect, const uint8_t* buffer) {
    int row_length = (rect.max.x - rect.min.x);
    for (int y = rect.min.y; y < rect.max.y; y++) {
        int cells_offset = y * wide_size + rect.min.x;
        int buffer_offset = (y - rect.min.y) * row_length;
        
        memcpy(cells + cells_offset, buffer + buffer_offset, row_length);
    }
}

static void debug_print_sub_region(uint8_t* cells, Rect rect) {
    int row_length = (rect.max.x - rect.min.x);
    for (int y = rect.min.y; y < rect.max.y; y++) {
        for (int x = rect.min.x; x < rect.max.x; x++) {
            uint8_t cell = cells[(y - rect.min.y) * row_length + x - rect.min.x];
            printf("%c", cell ? '#' : '.');
        }
        printf("\n");
    }
    printf("\n");
}


static Vec2i neighbour_tile(Vec2i tile, NeighbourIndex idx, const GridDimensions* gd) {
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

static void send_margins(int tx, int ty, uint8_t* dst, const GridDimensions* gd) {
    /* --- SEND MARGIN --- */
    int corner_area = gd->margin_width * gd->margin_width;
    int border_area = gd->tile_size * gd->margin_width;

    int margin_buffer_size = (corner_area > border_area) ? corner_area : border_area;
    
    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * margin_buffer_size);
    int my_index = index_of_tile(tx, ty, gd);

    int mw = gd->margin_width;
    int ts = gd->tile_size;
    Rect subregions[NEIGHBOUR_INDEX_COUNT] = {
        [TOPLEFT] = { mw, mw, 2 * mw, 2 * mw },
        [TOP] = { mw, mw, mw + ts, 2 * mw },
        [TOPRIGHT] = { ts, mw, ts + mw, 2 * mw },
        [LEFT] = { mw, mw, 2 * mw, mw + ts},
        [RIGHT] = { ts, mw, ts + mw, ts + mw},
        [BOTLEFT] = { mw, ts, 2 * mw, ts + mw},
        [BOT] = { mw, ts, ts + mw, ts + mw },
        [BOTRIGHT] = { ts, ts, ts + mw, ts + mw },
    };

    int subregion_sizes[NEIGHBOUR_INDEX_COUNT] = {
        [TOPLEFT] = corner_area,
        [TOP] = border_area,
        [TOPRIGHT] = corner_area,
        [LEFT] = border_area,
        [RIGHT] = border_area,
        [BOTLEFT] = corner_area,
        [BOT] = border_area,
        [BOTRIGHT] = corner_area,
    };

    const char* neighbour_labels[] = { "TOPLEFT", "TOP", "TOPRIGHT",
                                       "LEFT", "RIGHT",
                                       "BOTLEFT", "BOT", "BOTRIGHT"
    };
    
    for (NeighbourIndex i = 0; i < NEIGHBOUR_INDEX_COUNT; i++) {
        Vec2i nb = neighbour_tile((Vec2i){tx, ty}, i, gd);        
        pack_sub_grid(dst, gd->wide_size, subregions[i], margin_buffer);
        send_to_neighbour(margin_buffer,	
                          subregion_sizes[i],			
                          nb.x,
                          nb.y,
                          my_index,			
                          gd);
    }
    free(margin_buffer);
}

static void recv_margins(int tx, int ty, uint8_t* dst, const GridDimensions* gd) {
    /* --- SEND MARGIN --- */
    int corner_area = gd->margin_width * gd->margin_width;
    int border_area = gd->tile_size * gd->margin_width;

    int margin_buffer_size = (corner_area > border_area) ? corner_area : border_area;
    
    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * margin_buffer_size);
    int my_index = index_of_tile(tx, ty, gd);

    int mw = gd->margin_width;
    int ts = gd->tile_size;
    Rect subregions[NEIGHBOUR_INDEX_COUNT] = {
        [TOPLEFT] = { 0, 0, mw, mw },
        [TOP] = { mw, 0, mw + ts, mw },
        [TOPRIGHT] = { ts + mw, 0, ts + 2 * mw, mw },
        [LEFT] = { 0, mw, mw, mw + ts},
        [RIGHT] = { ts + mw, mw, ts + 2 * mw, ts + mw},
        [BOTLEFT] = { 0, ts + mw, mw, ts + 2 * mw},
        [BOT] = { mw, ts + mw, ts + mw, ts + 2 * mw },
        [BOTRIGHT] = { ts + mw, ts + mw, ts + 2 * mw, ts + 2 * mw },
    };

    int subregion_sizes[NEIGHBOUR_INDEX_COUNT] = {
        [TOPLEFT] = corner_area,
        [TOP] = border_area,
        [TOPRIGHT] = corner_area,
        [LEFT] = border_area,
        [RIGHT] = border_area,
        [BOTLEFT] = corner_area,
        [BOT] = border_area,
        [BOTRIGHT] = corner_area,
    };

    for (NeighbourIndex i = 0; i < NEIGHBOUR_INDEX_COUNT; i++) {
        Vec2i nb = neighbour_tile((Vec2i){tx, ty}, i, gd);
        recv_from_neighbour(margin_buffer,	
                            subregion_sizes[i],
                            nb.x,
                            nb.y,
                            my_index,			
                            gd);
        unpack_sub_grid(dst, gd->wide_size, subregions[i], margin_buffer);
    }
    free(margin_buffer);
}

static void worker(int rank, int iter, const GridDimensions* gd) {
    double tstart = MPI_Wtime();
    
    const Rect grid_center = {
        gd->margin_width,
        gd->margin_width,
        gd->tile_size + gd->margin_width,
        gd->tile_size + gd->margin_width
    };
    
    int tile_count;
    MPI_Recv(&tile_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    Tile* tiles = malloc(sizeof(Tile) * tile_count);
    uint8_t* cell_buffer = malloc(sizeof(uint8_t) * gd->wide_size * gd->wide_size * 2 * tile_count);
    
    for (int i = 0; i < tile_count; i++) {
        tiles[i].cells[0] = &cell_buffer[gd->wide_size * gd->wide_size * 2 * i];
        tiles[i].cells[1] = &cell_buffer[gd->wide_size * gd->wide_size * (2 * i + 1)];
    }

    uint8_t* recv_buffer = malloc(sizeof(uint8_t) * gd->tile_size * gd->tile_size);
    
    for (int i = 0; i < tile_count; i++) {
        MPI_Status status;
        MPI_Recv(recv_buffer, gd->tile_size*gd->tile_size, MPI_BYTE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        
        unpack_sub_grid(tiles[i].cells[0],
                        gd->wide_size,
                        grid_center,
                        recv_buffer);
        
        tile_of_index(status.MPI_TAG, &tiles[i].x, &tiles[i].y, gd);
    }

    free(recv_buffer);
    
    double tinit = MPI_Wtime();
    int src_index = 0;

    double tsend = 0.0, trecv = 0.0, tupdate = 0.0;

    int current_step = 0;
    while (current_step < iter) {
        double t0 = MPI_Wtime();
	
        for (int ti = 0; ti < tile_count; ti++) {
            send_margins(tiles[ti].x,
                         tiles[ti].y,
                         tiles[ti].cells[src_index],
                         gd);
        }
            
        double t1 = MPI_Wtime();
	
        for (int ti = 0; ti < tile_count; ti++) {
            recv_margins(tiles[ti].x,
                         tiles[ti].y,
                         tiles[ti].cells[src_index],
                         gd);
        }

        double t2 = MPI_Wtime();
	
        for (int growing_margin = 1;
             growing_margin <= gd->margin_width;
             growing_margin++) {
            for (int ti = 0; ti < tile_count; ti++) {
                update_tile_inside(tiles[ti].cells[src_index],
                                   tiles[ti].cells[!src_index],
                                   gd->wide_size,
                                   growing_margin);
            }
            src_index = !src_index;
            current_step++;
            if (current_step >= iter) {
                break;
            }
        }
	
        double t3 = MPI_Wtime();

        tsend += t1 - t0;
        trecv += t2 - t1;
        tupdate += t3 - t2;
    }

    double tfinish_iter = MPI_Wtime();
    
    uint8_t* send_buffer = malloc(sizeof(uint8_t) * gd->tile_size * gd->tile_size);

    for (int ti = 0; ti < tile_count; ti++) {
        pack_sub_grid(tiles[ti].cells[iter%2],
                      gd->wide_size,
                      grid_center,
                      send_buffer);
        int tag = index_of_tile(tiles[ti].x, tiles[ti].y, gd);
        MPI_Send(send_buffer, gd->tile_size * gd->tile_size, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
    }

    double tfinish_send = MPI_Wtime();

    free(send_buffer);
    free(cell_buffer);
    free(tiles);

    printf("[Worker %d] total time %gs, init %gs, loop : (send %gs, receive %gs, update %gs), finalize %gs\n",
           rank,
           tfinish_send - tstart,
           tinit - tstart,
           tsend,
           trecv,
           tupdate,
           tfinish_send - tfinish_iter);
}

void init_tiles_randomly(uint8_t *tiles, int total_cell_count, float density){
    for (int i = 0; i < total_cell_count; i++){
        tiles[i] = ((float) rand()) / RAND_MAX <= density;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, node_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count);

    /* Initialize the cells */
    Options options;
    if (!parse_options(&options, argc, argv)) {
        fprintf(stderr,
                "Usage :\n"
                "  game_of_life [-w <width>] [-h <height>] [-d <density>] [-t <tile size>] [-i <iter>] [-g] [-s <seed>] [-o <output_file>] [-f <input_filepath>]\n");
        return 1;
    }
    srand(options.seed);

    /* Init tiles and scatter them to worker nodes */
    assert(options.width % options.tile_size == 0);
    assert(options.height % options.tile_size == 0);

    GridDimensions gd = {
        .width = options.width,
        .height = options.height,
        .tile_hcount = options.width / options.tile_size,
        .tile_vcount = options.height / options.tile_size,
        .tile_size = options.tile_size,
        .wide_size = options.tile_size + 2 * options.margin_width,
        .margin_width = options.margin_width,

        .node_count = node_count,
    };
	
    if (rank == 0) {
        // Send everyone their tile count
        for (int i = 1; i < node_count; i++) {
            int tile_count = tile_count_of_rank(i, &gd);
            MPI_Send(&tile_count, 1, MPI_INT, i, 0,MPI_COMM_WORLD);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        int total_cell_count = options.width * options.height;
        uint8_t* tiles = malloc(sizeof(uint8_t) * total_cell_count);
        if (options.input_filepath) {
            parse_rle_file(options.input_filepath, tiles, &gd);
        } else {
            init_tiles_randomly(tiles, total_cell_count, options.density);
        }

        // send initial state of cells
        for (int i = 0; i < gd.tile_vcount*gd.tile_hcount; i++){
            MPI_Send(&tiles[i * options.tile_size * options.tile_size],
                     options.tile_size * options.tile_size,
                     MPI_BYTE,
                     rank_of_index(i, &gd),
                     i,
                     MPI_COMM_WORLD);
        }
	
        /* printf("\n--- DONE SENDING EVERYTHING ---\n"); */

        uint8_t *tmp_tile = malloc(sizeof(uint8_t) * options.tile_size*options.tile_size);
        for (int i = 0; i < gd.tile_vcount*gd.tile_hcount; i++) {
            MPI_Status status;
            MPI_Recv(tmp_tile, options.tile_size*options.tile_size, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            memcpy(&tiles[options.tile_size * options.tile_size * status.MPI_TAG], tmp_tile, options.tile_size * options.tile_size * sizeof(uint8_t));
        }

        if (options.output_filepath) {
            save_grid_to_png(tiles, options.output_filepath, &gd);
        } else {
            print_grid(tiles, &gd);
        }

        free(tmp_tile);
        free(tiles);
    } else{
        worker(rank, options.iter, &gd);
    }

    MPI_Finalize();    

}
