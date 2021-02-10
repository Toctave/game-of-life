#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    int tile_hcount;
    int tile_vcount;
    int tile_size;
    int wide_size; // == tile_size + 2

    int node_count;
} GridDimensions;

typedef struct {
    int x;
    int y;
} Vec2i;

typedef struct {
    int x;
    int y;
    uint8_t* cells[2];
} Tile;

static uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

static int index_of_tile(int tx, int ty, const GridDimensions* gd) {
    return ty * gd->tile_hcount + tx;
}

static void tile_of_index(int idx, int* tx, int* ty, const GridDimensions* gd) {
    *tx = idx % gd->tile_hcount;
    *ty = idx / gd->tile_hcount;
}

static int rank_of_index(int idx, const GridDimensions* gd) {
    return 1 + idx % (gd->node_count - 1); 
}

static int tile_count_of_rank(int rank, const GridDimensions* gd) {
    int worker_count = gd->node_count - 1;
    int total_count = gd->tile_hcount * gd->tile_vcount;
    
    int bonus = (rank - 1) < (total_count % worker_count);
    return total_count / worker_count + bonus;
}

static int rank_of_tile(int tx, int ty, const GridDimensions* gd) {
    int idx = index_of_tile(tx, ty, gd);
    return rank_of_index(idx, gd);
}

typedef enum {
    TOPLEFT, TOP, TOPRIGHT,
    LEFT, RIGHT,
    BOTLEFT, BOT, BOTRIGHT
} NeighbourIndex;
    
static void send_to_neighbour(uint8_t* buffer, size_t size, int dstx, int dsty, int my_index, const GridDimensions* gd) {
    int neighbour = rank_of_tile(dstx, dsty, gd);
    int tag = (my_index << 16) | index_of_tile(dstx, dsty, gd);
    // printf("sending tag %d to tile (%d, %d)\n", tag, dstx, dsty);
    MPI_Send(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD);
}

static void recv_from_neighbour(uint8_t* buffer, size_t size, int srcx, int srcy, int my_idx, const GridDimensions* gd) {
    int neighbour = rank_of_tile(srcx, srcy, gd);
    int tag = (index_of_tile(srcx, srcy, gd) << 16) | my_idx;

    // printf("receiving data from tile (%d, %d)\n", tag, srcx, srcy);
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

static void send_margins(int tx, int ty, uint8_t* dst, const GridDimensions* gd) {
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
        neighbour_tx[i] = (neighbour_tx[i] + gd->tile_hcount) % gd->tile_hcount;
        neighbour_ty[i] = (neighbour_ty[i] + gd->tile_vcount) % gd->tile_vcount;
    }

    /* --- SEND MARGIN --- */
    int my_index = index_of_tile(tx, ty, gd);

    /* Send top and bottom */
    uint8_t* top_margin = &dst[gd->wide_size + 1];

    send_to_neighbour(top_margin, gd->tile_size, neighbour_tx[TOP], neighbour_ty[TOP], my_index, gd);

    uint8_t* bot_margin = &dst[gd->wide_size * gd->tile_size + 1];

    send_to_neighbour(bot_margin, gd->tile_size, neighbour_tx[BOT], neighbour_ty[BOT], my_index, gd);
    
    /* Send left and right */
    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * gd->tile_size);

    /* LEFT */
    for (int y = 1; y < gd->wide_size - 1; y++) {
	margin_buffer[y-1] = dst[gd->wide_size * y + 1];
    }

    send_to_neighbour(margin_buffer, gd->tile_size, neighbour_tx[LEFT], neighbour_ty[LEFT], my_index, gd);

    /* RIGHT */
    for (int y = 1; y < gd->wide_size - 1; y++) {
	margin_buffer[y-1] = dst[gd->wide_size * y + gd->tile_size];
    }
    send_to_neighbour(margin_buffer, gd->tile_size, neighbour_tx[RIGHT], neighbour_ty[RIGHT], my_index, gd);
    
    /* Send diagonals */
    send_to_neighbour(&dst[gd->wide_size + 1],
		      1,
		      neighbour_tx[TOPLEFT],
		      neighbour_ty[TOPLEFT],
		      my_index,
		      gd);
    send_to_neighbour(&dst[gd->wide_size + gd->tile_size],
		      1,
		      neighbour_tx[TOPRIGHT],
		      neighbour_ty[TOPRIGHT],
		      my_index,
		      gd);
    send_to_neighbour(&dst[gd->wide_size * gd->tile_size + 1],
		      1,
		      neighbour_tx[BOTLEFT],
		      neighbour_ty[BOTLEFT],
		      my_index,
		      gd);
    send_to_neighbour(&dst[gd->wide_size * gd->tile_size + gd->tile_size],
		      1,
		      neighbour_tx[BOTRIGHT],
		      neighbour_ty[BOTRIGHT],
		      my_index,
		      gd);

    free(margin_buffer);
}

static void recv_margins(int tx, int ty, uint8_t* cells, const GridDimensions* gd) {
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
        neighbour_tx[i] = (neighbour_tx[i] + gd->tile_hcount) % gd->tile_hcount;
        neighbour_ty[i] = (neighbour_ty[i] + gd->tile_vcount) % gd->tile_vcount;
    }

    int my_index = index_of_tile(tx, ty, gd);

    /* --- RECEIVE BORDER --- */
    uint8_t* top_margin = &cells[1];
    uint8_t* bot_margin = &cells[gd->wide_size * (gd->wide_size - 1) + 1];

    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * gd->tile_size);
    
    /* Recv top and bottom */
    recv_from_neighbour(top_margin, gd->tile_size, neighbour_tx[TOP], neighbour_ty[TOP], my_index, gd);
    
    recv_from_neighbour(bot_margin, gd->tile_size, neighbour_tx[BOT], neighbour_ty[BOT], my_index, gd);

    /* Left and right */
    recv_from_neighbour(margin_buffer, gd->tile_size, neighbour_tx[LEFT], neighbour_ty[LEFT], my_index, gd);
    for (int y = 1; y < gd->wide_size - 1; y++) {
	cells[gd->wide_size * y] = margin_buffer[y-1];
    }

    recv_from_neighbour(margin_buffer, gd->tile_size, neighbour_tx[RIGHT], neighbour_ty[RIGHT], my_index, gd);
    for (int y = 1; y < gd->wide_size - 1; y++) {
	cells[gd->wide_size * y + gd->wide_size - 1] = margin_buffer[y-1];
    }

    /* Diagonals */
    recv_from_neighbour(&cells[0],
			1,
			neighbour_tx[TOPLEFT],
			neighbour_ty[TOPLEFT],
			my_index,
			gd);
    recv_from_neighbour(&cells[gd->wide_size - 1],
			1,
			neighbour_tx[TOPRIGHT],
			neighbour_ty[TOPRIGHT],
			my_index,
			gd);
    recv_from_neighbour(&cells[gd->wide_size * (gd->wide_size - 1)],
			1,
			neighbour_tx[BOTLEFT],
			neighbour_ty[BOTLEFT],
			my_index,
			gd);
    recv_from_neighbour(&cells[gd->wide_size * gd->wide_size - 1],
			1,
			neighbour_tx[BOTRIGHT],
			neighbour_ty[BOTRIGHT],
			my_index,
			gd);
    
    free(margin_buffer);
}

static void copy_narrow_buffer_to_tile(uint8_t* tile_cells, const uint8_t* buffer, int tile_size) {
    int wide_size = tile_size + 2;

    for (int y = 0; y < tile_size; y++) {
	memcpy(&tile_cells[(y + 1) * wide_size + 1],
	       &buffer[y * tile_size],
	       tile_size * sizeof(uint8_t));
    }
}

static void copy_tile_to_narrow_buffer(uint8_t* buffer, const uint8_t* tile_cells, int tile_size) {
    int wide_size = tile_size + 2;

    for (int y = 0; y < tile_size; y++) {
	memcpy(&buffer[y * tile_size],
	       &tile_cells[(y + 1) * wide_size + 1],
	       tile_size * sizeof(uint8_t));
    }
}

void print_state_of_tile(uint8_t* cells, int tile_size) {
    for (int y = 0; y < tile_size; y++) {
	for (int x = 0; x < tile_size; x++) {
	    if (cells[y * tile_size + x]) {
		printf("#");
	    } else {
		printf(".");
	    }
	}
	printf("\n");
    }
}

static void worker(int rank, int iter, const GridDimensions* gd) {
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
	copy_narrow_buffer_to_tile(tiles[i].cells[0], recv_buffer, gd->tile_size);

        tile_of_index(status.MPI_TAG, &tiles[i].x, &tiles[i].y, gd);
    }

    free(recv_buffer);
    
    for (int i = 0; i < iter; i++){
        int src_index = i % 2;
        int dst_index = (i + 1) % 2;
        
	for (int ti = 0; ti < tile_count; ti++) {
	    send_margins(tiles[ti].x,
			 tiles[ti].y,
			 tiles[ti].cells[src_index],
			 gd);
	}
            
	for (int ti = 0; ti < tile_count; ti++) {
	    recv_margins(tiles[ti].x,
			 tiles[ti].y,
			 tiles[ti].cells[src_index],
			 gd);
	}

	for (int ti = 0; ti < tile_count; ti++) {
            update_tile_inside(tiles[ti].cells[src_index],
			       tiles[ti].cells[dst_index],
			       gd->tile_size);
        }
    }

    uint8_t* send_buffer = malloc(sizeof(uint8_t) * gd->tile_size * gd->tile_size);

    for (int ti = 0; ti < tile_count; ti++) {
        copy_tile_to_narrow_buffer(send_buffer, tiles[ti].cells[iter%2], gd->tile_size);
        int tag = index_of_tile(tiles[ti].x, tiles[ti].y, gd);
        MPI_Send(send_buffer, gd->tile_size * gd->tile_size, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
    }

    free(send_buffer);
    free(cell_buffer);
    free(tiles);
}

typedef struct {
    int width;
    int height;
    float density;
    bool gui_on;
    int iter;
    int seed;
    char* input_filepath;
    char* output_filepath;
    int tile_size;
} Options;

bool parse_options(Options* options, int argc, char** argv) {
    int i = 1;

    options->width = 128;
    options->height = 128;
    options->iter = 100;
    options->density = .5;
    options->gui_on = false;
    options->seed = time(NULL);
    options->input_filepath = NULL;
    options->output_filepath = NULL;
    options->tile_size = 16;

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
	else if (!strcmp(arg, "-t")) {
	    if (!opt) {
		return false;
	    }

	    options->tile_size = atoi(opt);
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

	    options->density = atof(opt);
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
	else if (!strcmp(arg, "-f")) {
	    if (!opt) {
		return false;
	    }
	    
	    options->input_filepath = opt;
	    i++;
	}
	else if (!strcmp(arg, "-o")) {
	    if (!opt) {
		return false;
	    }
	    
	    options->output_filepath = opt;
	    i++;
	}
	else if (!strcmp(arg, "-s")) {
	    if (!opt) {
		return false;
	    }
	    options->seed = atoi(opt);
	    i++;
	}
	else {
	    return false;
	}
    }
    return true;
}

void init_tiles_randomly(uint8_t *tiles, int total_cell_count, float density){
    for (int i = 0; i < total_cell_count; i++){
        tiles[i] = ((float) rand()) / RAND_MAX <= density;
    }
}

void parse_rle_file(const char* filepath, uint8_t* tiles, const GridDimensions* gd) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
	fprintf(stderr, "could not open file '%s'\n", filepath);
	exit(1);
    }

    // cursor :
    int cx = 0;
    int cy = 0;

    char line_buffer[256];

    int width, height;
    fgets(line_buffer, 256, file);

    if (sscanf(line_buffer, "x = %d, y = %d", &width, &height) != 2) {
	fprintf(stderr, "Could not parse width and height of .rle pattern\n");
	exit(1);
    }
    
    while (fgets(line_buffer, 256, file)) {
	int count;
	char symbol;
	
	char* c = line_buffer;
	int chars_read;

	while (*c) {
	    chars_read = 0;
	    if (sscanf(c, "%d%n", &count, &chars_read) != 1) {
		count = 1;
	    }
	    c += chars_read;

	    sscanf(c, "%c%n", &symbol, &chars_read);
	    c += chars_read;

	    uint8_t cell_state;

	    switch (symbol) {
	    case 'b':
		cell_state = 0;
		break;
	    case 'o':
		cell_state = 1;
		break;
	    case '$':
		cy += count;
		cx = 0;
		continue;
	    case '\n':
		continue;
	    case '!':
		return;
	    default:
		break;
	    }

	    for (int x = cx; x < cx + count; x++) {
		int tx = x / gd->tile_size;
		int ty = cy / gd->tile_size;

		int ti = index_of_tile(tx, ty, gd);
		int dx = x % gd->tile_size;
		int dy = cy % gd->tile_size;

		tiles[ti * gd->tile_size * gd->tile_size + dy * gd->tile_size + dx] = cell_state;
	    }
	    cx += count;
	}
    }
}

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
} RGBAPixel;

void save_grid_to_png(const uint8_t* tiles, const char* filepath, const GridDimensions* gd) {
    RGBAPixel* png_data = malloc(sizeof(RGBAPixel) * gd->width * gd->height);
    for (int j = 0; j < gd->tile_vcount; j++) {
	for (int i = 0; i < gd->tile_hcount; i++) {
	    const uint8_t* tile = &tiles[gd->tile_size * gd->tile_size * (j * gd->tile_hcount + i)];
	    for (int y = 0; y < gd->tile_size; y++) {
		const uint8_t* tile_row = &tile[y * gd->tile_size];
		for (int x = 0; x < gd->tile_size; x++) {
		    int pngy = j * gd->tile_size + y;
		    int pngx = i * gd->tile_size + x;
		    int pngi = pngy * gd->width + pngx;
			    
		    if (tile_row[x]) {
			png_data[pngi] = (RGBAPixel){255, 255, 255, 255};
		    } else if ((i + j) % 2 == 0) {
			png_data[pngi] = (RGBAPixel){64, 64, 64, 255};
		    } else {
			png_data[pngi] = (RGBAPixel){32, 32, 32, 255};
		    }
		}
	    }
	}
    }

    printf("Writing png '%s'\n", filepath);
    stbi_write_png(filepath, gd->width, gd->height, 4, png_data, gd->width * sizeof(RGBAPixel));
    free(png_data);
}

void print_grid(const uint8_t* tiles, const GridDimensions* gd) {
    for (int j = 0; j < gd->tile_vcount; j++) {
	for (int y = 0; y < gd->tile_size; y++) {
	    for (int i = 0; i < gd->tile_hcount; i++) {
		const uint8_t* tile_row = &tiles[gd->tile_size * gd->tile_size * (j * gd->tile_hcount + i)
						 + y * gd->tile_size];
		for (int x = 0; x < gd->tile_size; x++) {
		    if (tile_row[x]) {
			printf("#");
		    } else {
			printf(".");
		    }
		}
		printf(" ");
	    }
	    printf("\n");
	}
	printf("\n");
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
	.wide_size = options.tile_size + 2,

	.node_count = node_count
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
