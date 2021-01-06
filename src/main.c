#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#include <SDL2/SDL.h>

typedef struct {
    int width;
    int height;
    int marginx;
    int marginy;
    uint8_t* cells[2];
    int iterations;
    bool stop;
} GolData;

static void wipe_surface(SDL_Surface* surface) {
    memset(surface->pixels, 0, surface->pitch * surface->h);
}

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

static void render_cells(SDL_Surface* surface, GolData* gol) {
    uint8_t* cells = gol->cells[gol->iterations%2];
    for (int y = 0; y < gol->height - gol->marginy * 2; y++) {
	for (int x = 0; x < gol->width - gol->marginx * 2; x++) {
	    char* pixel = (char*) surface->pixels + y * surface->pitch + x * surface->format->BytesPerPixel;
	    uint32_t pixel_value;
	    if (cells[(y + gol->marginy) * gol->width + x + gol->marginx]) {
		pixel_value = SDL_MapRGB(surface->format, 255, 255, 255);
	    } else {
		pixel_value = SDL_MapRGB(surface->format, 0, 0, 0);
	    }
	    *(uint32_t*)pixel = pixel_value;
	}
    }
}

static int iterate_loop(void* data) {
    GolData* gol = (GolData*) data;
    while (true) {
	update_cells(gol);
	gol->iterations++;

	if (gol->stop)
	    return 0;
    }
    return 0;
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

int main(int argc, char** argv) {
    if (argc != 4) {
	fprintf(stderr, "Usage :\n  game_of_life <width> <height> <initial proportion of live cells>\n");
	return 1;
    }
    
    SDL_Init(SDL_INIT_VIDEO);
    srand(time(NULL));

    GolData gol;
    gol.marginx = 1;
    gol.marginy = 1;
    
    gol.width = atoi(argv[1]) + gol.marginx * 2;
    gol.width = gol.width - (gol.width % gol.marginx); // make it a multiple of the margin
    
    gol.height = atoi(argv[2]) + gol.marginy * 2;
    gol.height = gol.height - (gol.height % gol.marginy); // make it a multiple of the margin
    
    
    initialize_gol(&gol);
    fill_random(&gol, atof(argv[3]));
    
    SDL_Window* window = SDL_CreateWindow("Game of life",
    					  SDL_WINDOWPOS_CENTERED,
    					  SDL_WINDOWPOS_CENTERED,
    					  gol.width - gol.marginx * 2,
    					  gol.height - gol.marginy * 2,
    					  0);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    wipe_surface(screen);

    SDL_Thread* iterate_thread = SDL_CreateThread(iterate_loop, "iterate", &gol);

    int t0 = SDL_GetTicks();
    
    while (!gol.stop) {
    	SDL_Event evt;
    	while (SDL_PollEvent(&evt)) {
    	    if (evt.type == SDL_QUIT ||
		(evt.type == SDL_KEYDOWN && evt.key.keysym.sym == SDLK_ESCAPE)) {
    		gol.stop = true;
    	    }
    	}
	
    	render_cells(screen, &gol);
    	SDL_UpdateWindowSurface(window);
    }

    SDL_WaitThread(iterate_thread, NULL);
    
    int t1 = SDL_GetTicks();
    double time_per_iteration = (double) (t1 - t0) / ((double)gol.iterations * gol.width * gol.height);
    printf("%d iterations in %dms, %.3fns per cell per iteration\n", gol.iterations, t1 - t0, time_per_iteration * 1000. * 1000.);

    free_gol(&gol);
    SDL_Quit();
    return 0;
}
