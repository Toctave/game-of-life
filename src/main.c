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
    return 0; // TODO
}

static int rank_of_tile(int tx, int ty, int node_count, int tile_hcount) {
    int idx = index_of_tile(tx, ty, tile_hcount);
    return rank_of_index(idx, node_count, tile_hcount);
}

static void update_tile(int tx, int ty, uint8_t* src, uint8_t* dst, int tile_size, int node_count, int tile_hcount) {
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

    /* --- SEND BORDER --- */
    int my_index = index_of_tile(tx, ty, tile_hcount);
    int neighbour;
    /* Send top and bottom */
    uint8_t* top_margin = &dst[wide_size + 1];
    neighbour = rank_of_tile(tx, ty - 1, node_count, tile_hcount);
    MPI_Send(top_margin, tile_size, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    uint8_t* bottom_margin = &dst[wide_size * tile_size + 1];
    neighbour = rank_of_tile(tx, ty + 1, node_count, tile_hcount);
    MPI_Send(bottom_margin, tile_size, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);
    
    /* Send left and right */
    int* margin_buffer = malloc(sizeof(int) * tile_size);

    /* LEFT */
    for (int y = 1; y < wide_size - 1; y++) {
	margin_buffer[y-1] = dst[wide_size * y + 1];
    }
    neighbour = rank_of_tile(tx - 1, ty, node_count, tile_hcount);
    MPI_Send(margin_buffer, tile_size, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    /* RIGHT */
    for (int y = 1; y < wide_size - 1; y++) {
	margin_buffer[y-1] = dst[wide_size * y + tile_size];
    }
    neighbour = rank_of_tile(tx + 1, ty, node_count, tile_hcount);
    MPI_Send(margin_buffer, tile_size, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    /* Send diagonals */
    neighbour = rank_of_tile(tx - 1, ty - 1, node_count, tile_hcount);
    MPI_Send(&dst[wide_size + 1], 1, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);
    
    neighbour = rank_of_tile(tx + 1, ty - 1, node_count, tile_hcount);
    MPI_Send(&dst[wide_size + tile_size], 1, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    neighbour = rank_of_tile(tx - 1, ty + 1, node_count, tile_hcount);
    MPI_Send(&dst[wide_size * tile_size + 1], 1, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    neighbour = rank_of_tile(tx + 1, ty + 1, node_count, tile_hcount);
    MPI_Send(&dst[wide_size * tile_size + tile_size], 1, MPI_BYTE, neighbour, my_index, MPI_COMM_WORLD);

    /* --- RECEIVE BORDER --- */
    /* Recv top and bottom */
    neighbour = rank_of_tile(tx, ty - 1, node_count, tile_hcount);    
    MPI_Recv(top_margin, tile_size, MPI_BYTE, neighbour, index_of_tile(tx, ty - 1, tile_hcount), MPI_COMM_WORLD, NULL);

    neighbour = rank_of_tile(tx, ty + 1, node_count, tile_hcount);
    MPI_Recv(bottom_margin, tile_size, MPI_BYTE, neighbour, index_of_tile(tx, ty + 1, tile_hcount), MPI_COMM_WORLD, NULL);

    /* Left and right */
    neighbour = rank_of_tile(tx - 1, ty, node_count, tile_hcount);
    MPI_Recv(margin_buffer, tile_size, MPI_BYTE, neighbour, index_of_tile(tx - 1, ty, tile_hcount), MPI_COMM_WORLD, NULL);
    for (int y = 1; y < wide_size - 1; y++) {
	dst[wide_size * y + 1] = margin_buffer[y-1];
    }

    neighbour = rank_of_tile(tx + 1, ty, node_count, tile_hcount);
    MPI_Recv(margin_buffer, tile_size, MPI_BYTE, neighbour, index_of_tile(tx + 1, ty, tile_hcount), MPI_COMM_WORLD, NULL);
    for (int y = 1; y < wide_size - 1; y++) {
	dst[wide_size * y + tile_size] = margin_buffer[y-1];
    }

    /* Diagonals */
    neighbour = rank_of_tile(tx - 1, ty - 1, node_count, tile_hcount);
    MPI_Recv(&dst[wide_size + 1], 1, MPI_BYTE, neighbour, index_of_tile(tx - 1, ty - 1, tile_hcount), MPI_COMM_WORLD, NULL);
    
    neighbour = rank_of_tile(tx + 1, ty - 1, node_count, tile_hcount);
    MPI_Recv(&dst[wide_size + tile_size], 1, MPI_BYTE, neighbour, index_of_tile(tx + 1, ty - 1, tile_hcount), MPI_COMM_WORLD, NULL);

    neighbour = rank_of_tile(tx - 1, ty + 1, node_count, tile_hcount);
    MPI_Recv(&dst[wide_size * tile_size + 1], 1, MPI_BYTE, neighbour, index_of_tile(tx - 1, ty + 1, tile_hcount), MPI_COMM_WORLD, NULL);

    neighbour = rank_of_tile(tx + 1, ty + 1, node_count, tile_hcount);
    MPI_Recv(&dst[wide_size * tile_size + tile_size], 1, MPI_BYTE, neighbour, index_of_tile(tx + 1, ty + 1, tile_hcount), MPI_COMM_WORLD, NULL);
    
    free(margin_buffer);
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
} Options;

bool parse_options(Options* options, int argc, char** argv) {
    int i = 1;

    options->width = 128;
    options->height = 128;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, node_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count);

    int tile_size = 64;
    
    if (rank == 0) {
	/* Initialize the cells */
	Options options;
	if (!parse_options(&options, argc, argv)) {
	    fprintf(stderr,
		    "Usage :\n"
		    "  game_of_life [-w <width>] [-h <height>] [-d <density>] [-g]\n");
	    return 1;
	}
	srand(time(NULL));
	
	GolData gol;
	gol.width = options.width;
    	gol.height = options.height;
    
	initialize_gol(&gol);
	fill_random(&gol, options.density);

	/* Scatter them to worker nodes */
	assert(gol.width % tile_size == 0);
	assert(gol.height % tile_size == 0);

	
    } else {

    }

    MPI_Finalize();    
}
