#include "io.h"

#include "tile_indexing.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

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
    options->margin_width = 1;

    MPI_Comm_size(MPI_COMM_WORLD, &options->node_count);
    options->thread_count = omp_get_num_threads();

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
	else if (!strcmp(arg, "--margin-width")) {
	    if (!opt) {
		return false;
	    }
	    options->margin_width = atoi(opt);
	    i++;
	}
	else {
	    return false;
	}
    }

    char log_file_start[256];

    if (options->input_filepath) {
        sprintf(log_file_start, "f_%s", options->input_filepath);
    } else {
        sprintf(log_file_start, "rnd_%.3f", options->density);
    }

    sprintf(options->log_filepath,
            "logs/%s_nodes%d_threads%d_w%d_h%d_ts%d_mw%d.txt",
            log_file_start,
	    options->node_count,
	    options->thread_count,
            options->width,
            options->height,
            options->tile_size,
            options->margin_width);

    for (char* s = options->log_filepath + strlen("logs/"); *s; s++) {
        if (*s == '/')
            *s = '.';
    }
    
    return true;
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
		return;
	    }

	    for (int x = cx; x < cx + count; x++) {
		int tx = x / gd->tile_size;
		int ty = cy / gd->tile_size;

		int ti = index_of_tile((Vec2i){tx, ty}, gd);
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

