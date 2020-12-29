#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include <SDL2/SDL.h>

static void wipe_surface(SDL_Surface* surface) {
    memset(surface->pixels, 0, surface->pitch * surface->h);
}

static uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

static void copy_redundant_borders(uint8_t* cells, int width, int height) {
    for (int i = 1; i < width - 1; i++) {
	cells[i] =
	    cells[(height - 2) * width + i];
	cells[(height - 1) * width + i] =
	    cells[width + i];
    }
    
    for (int j = 0; j < height; j++) {
	cells[j * width] =
	    cells[(j + 1) * width - 2];
	cells[(j + 1) * width - 1] =
	    cells[j * width + 1];
    }
}

static void update_cells(uint8_t* src, uint8_t* dst, int width, int height) {
    int neighbour_offsets[8] = {
	-width - 1, -width, -width + 1,
	        -1,                  1,
	 width - 1,  width,  width + 1
    };
    for (int j = 1; j < height - 1; j++) {
	for (int i = 1; i < width - 1; i++) {
	    int neighbours = 0;
	    int base = j * width + i;
	    for (int k = 0; k < 8; k++) {
		neighbours += src[base + neighbour_offsets[k]];
	    }
	    dst[base] = rule(src[base], neighbours);
	}
    }

    copy_redundant_borders(dst, width, height);
}

static void render_cells(SDL_Surface* surface, uint8_t* cells, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
	for (int x = 1; x < width - 1; x++) {
	    char* pixel = (char*) surface->pixels + y * surface->pitch + x * surface->format->BytesPerPixel;
	    uint32_t pixel_value;
	    if (cells[y * width + x]) {
		pixel_value = SDL_MapRGB(surface->format, 255, 255, 255);
	    } else {
		pixel_value = SDL_MapRGB(surface->format, 0, 0, 0);
	    }
	    *(uint32_t*)pixel = pixel_value;
	}
    }
}

typedef struct {
    int width;
    int height;
    uint8_t* cells[2];
    int iterations;
    bool stop;
} GolData;

static int iterate_loop(void* data) {
    GolData* gol = (GolData*) data;
    while (true) {
	update_cells(gol->cells[gol->iterations%2],
		     gol->cells[(gol->iterations+1)%2],
		     gol->width,
		     gol->height);
	gol->iterations++;

	if (gol->stop)
	    return 0;
    }
    return 0;
}

static void initialize_gol(GolData* gol, float density) {
    gol->cells[0] = malloc(gol->width * gol->height * 2);
    gol->cells[1] = gol->cells[0] + gol->width * gol->height;

    for (int y = 0; y < gol->height; y++) {
	for (int x = 0; x < gol->width; x++) {
	    gol->cells[0][y * gol->width + x] = (float) rand() / RAND_MAX <= density;
	}
    }

    copy_redundant_borders(gol->cells[0], gol->width, gol->height);
    gol->iterations = 0;
    gol->stop = false;
}

static void free_gol(GolData* gol) {
    free(gol->cells[0]);
}

int main(int argc, char** argv) {
    if (argc != 4) {
	return 1;
    }
    
    srand(time(NULL));

    GolData gol;
    gol.width = atoi(argv[1]);
    gol.height = atoi(argv[2]);
    initialize_gol(&gol, atof(argv[3]));
    
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Game of life",
					  SDL_WINDOWPOS_CENTERED,
					  SDL_WINDOWPOS_CENTERED,
					  gol.width, gol.height,
					  0);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    wipe_surface(screen);

    SDL_Thread* iterate_thread = SDL_CreateThread(iterate_loop, "iterate", &gol);

    int t0 = SDL_GetTicks();
    
    while (!gol.stop) {
	SDL_Event evt;
	while (SDL_PollEvent(&evt)) {
	    if (evt.type == SDL_QUIT) {
		gol.stop = true;
	    }
	}
	
	render_cells(screen, gol.cells[(gol.iterations+1)%2], gol.width, gol.height);
	SDL_UpdateWindowSurface(window);
    }

    SDL_WaitThread(iterate_thread, NULL);
    
    int t1 = SDL_GetTicks();
    float time_per_iteration = (float) (t1 - t0) / gol.iterations;
    printf("%d iterations in %dms, %fms per iteration\n", gol.iterations, t1 - t0, time_per_iteration);

    free_gol(&gol);
    SDL_Quit();
    return 0;
}
