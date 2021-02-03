#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include <mpi.h>

typedef struct {
    int width;
    int height;
    int marginx;
    int marginy;
    uint8_t* cells[2];
    int iterations;
    bool stop;
} GolData;

static uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

static void copy_redundant_borders(GolData* gol) {
    uint8_t* dst = gol->cells[(gol->iterations + 1) % 2];
    
    for (int i = gol->marginx; i < gol->width - gol->marginx; i++) {
	for (int j = 0; j < gol->marginy; j++) {
	    dst[j * gol->width + i] =
		
		dst[(gol->height - 2 * gol->marginy + j) * gol->width + i];
	    dst[(gol->height - gol->marginy + j) * gol->width + i] =
		dst[(gol->marginy + j) * gol->width + i];
	}
    }
    
    for (int j = 0; j < gol->height; j++) {
	memcpy(
	    &dst[j * gol->width],
	    &dst[(j + 1) * gol->width - 2 * gol->marginx],
	    gol->marginx
	    );
	memcpy(
	    &dst[(j + 1) * gol->width - gol->marginx],
	    &dst[j * gol->width + gol->marginx],
	    gol->marginx
	    );
    }
}

static void update_cells(GolData* gol) {
    uint8_t* src = gol->cells[gol->iterations % 2];
    uint8_t* dst = gol->cells[(gol->iterations + 1) % 2];
    
    int neighbour_offsets[8] = {
	-gol->width - 1, -gol->width, -gol->width + 1,
	-1,                  1,
	gol->width - 1,  gol->width,  gol->width + 1
    };
    for (int j = 1; j < gol->height - 1; j++) {
	for (int i = 1; i < gol->width - 1; i++) {
	    int neighbours = 0;
	    int base = j * gol->width + i;
	    for (int k = 0; k < 8; k++) {
		neighbours += src[base + neighbour_offsets[k]];
	    }
	    dst[base] = rule(src[base], neighbours);
	}
    }

    copy_redundant_borders(gol);
}

static int index_of_tile(int tx, int ty, int tile_hcount) {
    return ty * tile_hcount + tx;
}

static void tile_of_index(int idx, int tile_hcount, int* tx, int* ty) {
    *tx = idx % tile_hcount;
    *ty = idx / tile_hcount;
}

static int rank_of_index(int idx, int node_count, int tile_hcount) {
    return 1; // TODO
}

static int tile_count_of_rank(int rank, int tile_count) { 
    // TODO
    if (rank == 1) {
        return tile_count;
    } else {
        return 0;
    }
}

static int rank_of_tile(int tx, int ty, int node_count, int tile_hcount) {
    int idx = index_of_tile(tx, ty, tile_hcount);
    return rank_of_index(idx, node_count, tile_hcount);
}

typedef enum {
    TOPLEFT, TOP, TOPRIGHT,
    LEFT, RIGHT,
    BOTLEFT, BOT, BOTRIGHT
} NeighbourIndex;
    
static void send_to_neighbour(uint8_t* buffer, size_t size, int dstx, int dsty, int tag, int node_count, int tile_hcount) {
    int neighbour = rank_of_tile(dstx, dsty, node_count, tile_hcount);
    printf("sending tag %d to tile (%d, %d)\n", tag, dstx, dsty);
    MPI_Send(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD);
}

static void recv_from_neighbour(uint8_t* buffer, size_t size, int srcx, int srcy, int node_count, int tile_hcount) {
    int neighbour = rank_of_tile(srcx, srcy, node_count, tile_hcount);
    int tag = index_of_tile(srcx, srcy, tile_hcount);
    printf("receiving tag %d from tile (%d, %d)\n", tag, srcx, srcy);
    MPI_Recv(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD, NULL);
}

static void update_tile_inside(uint8_t* src, uint8_t* dst, int tile_size) {
    int wide_size = tile_size + 2;

    int neighbour_offsets[8] = {
	-wide_size - 1, -wide_size, -wide_size + 1,
	-1,                                      1,
	wide_size - 1,   wide_size,  wide_size + 1
    };

    /* Update internal tile */
    for (int j = 1; j < wide_size - 1; j++) {
	    for (int i = 1; i < wide_size - 1; i++) {
	        int neighbours = 0;
	        int base = j * wide_size + i;
	        for (int k = 0; k < 8; k++) {
		        neighbours += src[base + neighbour_offsets[k]];
	        }
	        dst[base] = rule(src[base], neighbours);
	    }
    }  
}  
 
static void send_margins(int tx, int ty, uint8_t* dst, int tile_size, int node_count, int tile_hcount, int tile_vcount) {
    int neighbour_tx[8] = { 
        tx - 1, tx, tx + 1,
        tx - 1,     tx + 1, 
        tx - 1, tx, tx + 1,
    };
    
    int neighbour_ty[8] = { 
        ty - 1, ty - 1, ty - 1,
        ty,             ty,
        ty + 1, ty + 1, ty + 1,
    };
    
    for (int i = 0; i < 8; i++) {
        neighbour_tx[i] = (neighbour_tx[i] + tile_hcount) % tile_hcount;
        neighbour_ty[i] = (neighbour_ty[i] + tile_vcount) % tile_vcount;
    }

    /* --- SEND BORDER --- */
    int my_index = index_of_tile(tx, ty, tile_hcount);
    printf("my index is %d\n", my_index);
    /* Send top and bottom */
    uint8_t* top_margin = &dst[wide_size + 1];
    send_to_neighbour(top_margin, tile_size, neighbour_tx[TOP], neighbour_ty[TOP], my_index, node_count, tile_hcount);

    uint8_t* bot_margin = &dst[wide_size * tile_size + 1];
    send_to_neighbour(bot_margin, tile_size, neighbour_tx[BOT], neighbour_ty[BOT], my_index, node_count, tile_hcount);
    
    /* Send left and right */
    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * tile_size);

    /* LEFT */
    for (int y = 1; y < wide_size - 1; y++) {
	margin_buffer[y-1] = dst[wide_size * y + 1];
    }
    send_to_neighbour(margin_buffer, tile_size, neighbour_tx[LEFT], neighbour_ty[LEFT], my_index, node_count, tile_hcount);

    /* RIGHT */
    for (int y = 1; y < wide_size - 1; y++) {
	margin_buffer[y-1] = dst[wide_size * y + tile_size];
    }
    send_to_neighbour(margin_buffer, tile_size, neighbour_tx[RIGHT], neighbour_ty[RIGHT], my_index, node_count, tile_hcount);
    
    /* Send diagonals */
    send_to_neighbour(&dst[wide_size + 1], 1, neighbour_tx[TOPLEFT], neighbour_ty[TOPLEFT], my_index, node_count, tile_hcount);
    send_to_neighbour(&dst[wide_size + tile_size], 1, neighbour_tx[TOPRIGHT], neighbour_ty[TOPRIGHT], my_index, node_count, tile_hcount);
    send_to_neighbour(&dst[wide_size * tile_size + 1], 1, neighbour_tx[BOTLEFT], neighbour_ty[BOTLEFT], my_index, node_count, tile_hcount);
    send_to_neighbour(&dst[wide_size * tile_size + tile_size], 1, neighbour_tx[BOTRIGHT], neighbour_ty[BOTRIGHT], my_index, node_count, tile_hcount);
}

static void recv_margins(int tx, int ty, uint8_t* dst, int tile_size, int node_count, int tile_hcount, int tile_vcount) {
    int neighbour_tx[8] = { 
        tx - 1, tx, tx + 1,
        tx - 1,     tx + 1, 
        tx - 1, tx, tx + 1,
    };
    
    int neighbour_ty[8] = { 
        ty - 1, ty - 1, ty - 1,
        ty,             ty,
        ty + 1, ty + 1, ty + 1,
    };
    
    for (int i = 0; i < 8; i++) {
        neighbour_tx[i] = (neighbour_tx[i] + tile_hcount) % tile_hcount;
        neighbour_ty[i] = (neighbour_ty[i] + tile_vcount) % tile_vcount;
    }

    /* --- RECEIVE BORDER --- */
    /* Recv top and bottom */
    recv_from_neighbour(top_margin, tile_size, neighbour_tx[TOP], neighbour_ty[TOP], node_count, tile_hcount);
    recv_from_neighbour(bot_margin, tile_size, neighbour_tx[BOT], neighbour_ty[BOT], node_count, tile_hcount);

    /* Left and right */
    recv_from_neighbour(margin_buffer, tile_size, neighbour_tx[LEFT], neighbour_ty[LEFT], node_count, tile_hcount);
    for (int y = 1; y < wide_size - 1; y++) {
	dst[wide_size * y + 1] = margin_buffer[y-1];
    }

    recv_from_neighbour(margin_buffer, tile_size, neighbour_tx[RIGHT], neighbour_ty[RIGHT], node_count, tile_hcount);
    for (int y = 1; y < wide_size - 1; y++) {
	dst[wide_size * y + tile_size] = margin_buffer[y-1];
    }

    /* Diagonals */
    recv_from_neighbour(&dst[wide_size + 1], 1, neighbour_tx[TOPLEFT], neighbour_ty[TOPLEFT], node_count, tile_hcount);
    recv_from_neighbour(&dst[wide_size + tile_size], 1, neighbour_tx[TOPRIGHT], neighbour_ty[TOPRIGHT], node_count, tile_hcount);
    recv_from_neighbour(&dst[wide_size * tile_size + 1], 1, neighbour_tx[BOTLEFT], neighbour_ty[BOTLEFT], node_count, tile_hcount);
    recv_from_neighbour(&dst[wide_size * tile_size + tile_size], 1, neighbour_tx[BOTRIGHT], neighbour_ty[BOTRIGHT], node_count, tile_hcount);
    
    free(margin_buffer);
}

typedef struct {
    int x;
    int y;
    uint8_t* cells[2];
} Tile;

static void worker(int tx, int ty, int tile_size, int node_count, int tile_hcount, int tile_vcount, int iter) {
    int tile_count;
    MPI_Recv(&tile_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    // TODO : allocate larger tiles and receive only into the inside part + loop send/recv/update

    Tile* tiles = malloc(sizeof(Tile) * tile_count);
    uint8_t* cell_buffer = malloc(sizeof(uint8_t) * tile_size * tile_size * 2 * tile_count);
    
    for (int i = 0; i < tile_count; i++) {
        tiles[i].cells[0] = &cell_buffer[tile_size * tile_size * 2 * i];
        tiles[i].cells[1] = &cell_buffer[tile_size * tile_size * (2 * i + 1)];
    }
    
    for (int i = 0; i < tile_count; i++) {
        MPI_Status status;
        MPI_Recv(tiles[i].cells[0], tile_size*tile_size, MPI_BYTE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        
        tile_of_index(status.MPI_TAG, tile_hcount, &tiles[i].x, &tiles[i].y);
    }
    
    
    for (int i = 0; i< iter; i++){
        for (int ti = 0; ti < tile_count; ti++) {
            update_tile_inside(uint8_t* src, uint8_t* dst, int tile_size);
            send_margins(int tx, int ty, uint8_t* dst, int tile_size, int node_count, int tile_hcount, int tile_vcount);
        }
        
        for (int ti = 0; ti < tile_count; ti++) {
            recv_margins(int tx, int ty, uint8_t* dst, int tile_size, int node_count, int tile_hcount, int tile_vcount);
        }
        update_tile(tx, ty, cells[i%2], cells[(i+1)%2], tile_size, node_count, tile_hcount, tile_vcount);
    }
    
    for (int y = 0; y < tile_size; y++) {
	    for (int x = 0; x < tile_size; x++) {
	        if (cells[iter%2][y * tile_size + x]) {
		        printf("#");
	        } else {
		        printf(".");
	        }
	    }
	    printf("\n");
    }
    
    //TODO send
    free(cells[0]);
}

static void dump_cells(FILE* out, GolData* gol) {
    uint8_t* cells = gol->cells[gol->iterations%2];
    for (int y = 0; y < gol->height - gol->marginy * 2; y++) {
	for (int x = 0; x < gol->width - gol->marginx * 2; x++) {
	    if (cells[(y + gol->marginy) * gol->width + x + gol->marginx]) {
		fprintf(out, "#");
	    } else {
		fprintf(out, ".");
	    }
	}
	fprintf(out, "\n");
    }
}

static void initialize_gol(GolData* gol) {
    gol->cells[0] = malloc(gol->width * gol->height * 2);
    gol->cells[1] = gol->cells[0] + gol->width * gol->height;

    for (int y = 0; y < gol->height; y++) {
	for (int x = 0; x < gol->width; x++) {
	    gol->cells[0][y * gol->width + x] = 0;
	}
    }

    gol->iterations = 0;
    gol->stop = false;
}

static void fill_random(GolData* gol, float density) {
    for (int y = 0; y < gol->height; y++) {
	for (int x = 0; x < gol->width; x++) {
	    gol->cells[0][y * gol->width + x] =
		((float) rand() / RAND_MAX) <= density;
	}
    }
}

static void free_gol(GolData* gol) {
    free(gol->cells[0]);
}

typedef struct {
    int width;
    int height;
    float density;
    bool gui_on;
    int iter;
} Options;

bool parse_options(Options* options, int argc, char** argv) {
    int i = 1;

    options->width = 128;
    options->height = 128;
    options->iter = 100;
    options->density = .5;
    options->gui_on = false;

    while (i < argc) {
	char* arg = argv[i];
	i++;
	
	char* opt = NULL;
	if (i < argc) {
	    opt = argv[i];
	}
	
	if (!strcmp(arg, "-w")) {
	    if (!opt) {
		return false;
	    }

	    options->width = atoi(opt);
	    i++;
	}
	else if (!strcmp(arg, "-h")) {
	    if (!opt) {
		return false;
	    }

	    options->height = atoi(opt);
	    i++;
	}
	else if (!strcmp(arg, "-d")) {
	    if (!opt) {
		return false;
	    }

	    options->density = atoi(opt);
	    i++;
	}
	else if (!strcmp(arg, "-i")) {
	    if (!opt) {
		return false;
	    }

	    options->iter = atoi(opt);
	    i++;
	}
	else if (!strcmp(arg, "-g")) {
	    options->gui_on = true;
	}
	else {
	    return false;
	}
    }
    return true;
}

int main_old(int argc, char** argv) {
    Options options;
    if (!parse_options(&options, argc, argv)) {
	fprintf(stderr,
		"Usage :\n"
		"  game_of_life [-w <width>] [-h <height>] [-d <density>] [-g]\n");
	return 1;
    }
    srand(time(NULL));

    GolData gol;
    gol.marginx = 1;
    gol.marginy = 1;
    
    gol.width = options.width + gol.marginx * 2;
    gol.width = gol.width - (gol.width % gol.marginx); // make it a multiple of the margin
    
    gol.height = options.height + gol.marginy * 2;
    gol.height = gol.height - (gol.height % gol.marginy); // make it a multiple of the margin
    
    initialize_gol(&gol);
    fill_random(&gol, options.density);

    while (!gol.stop) {
	update_cells(&gol);
	gol.iterations++;
    }

    dump_cells(stdout, &gol);
    
    /* printf("%d iterations in %dms, %.3fns per cell per iteration\n", gol.iterations, t1 - t0, time_per_iteration * 1000. * 1000.); */

    free_gol(&gol);
    
    return 0;
}

void init_tile(uint8_t *tile, int tile_size, float density){
    for (int i = 0; i < tile_size*tile_size; i++){
        tile[i] = ((float) rand() / RAND_MAX) <= density;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, node_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count);

    int tile_size = 64;
    
    /* Initialize the cells */
	Options options;
	if (!parse_options(&options, argc, argv)) {
	    fprintf(stderr,
		    "Usage :\n"
		    "  game_of_life [-w <width>] [-h <height>] [-d <density>] [-i <iter>] [-g]\n");
	    return 1;
	}
	srand(time(NULL));

	/* Init tiles and scatter them to worker nodes */
	assert(options.width % tile_size == 0);
	assert(options.height % tile_size == 0);
	
	int tile_hcount = options.width / tile_size;
	int tile_vcount = options.height / tile_size;
    
    if (rank == 0) {
        // Send everyone their tile count
        for (int i = 1; i < node_count; i++) {
            int tile_count = tile_count_of_rank(i, tile_hcount * tile_vcount);
	        MPI_Send(&tile_count, 1, MPI_INT, i, 0,MPI_COMM_WORLD);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
	
	    for (int i = 0; i < tile_vcount*tile_hcount; i++){
	            uint8_t *tile = malloc(sizeof(uint8_t) * tile_size*tile_size);
	            init_tile(tile, tile_size,options.density);
	            MPI_Send(tile,tile_size*tile_size, MPI_BYTE, rank_of_index(i,node_count,tile_hcount),i,MPI_COMM_WORLD);
	    }
	
    } else {
    
        int tx, ty;
        tile_of_index(rank - 1, tile_hcount, &tx, &ty);
        worker(tx, ty, tile_size, node_count, tile_hcount, tile_vcount, options.iter);

    }

    MPI_Finalize();    
}
