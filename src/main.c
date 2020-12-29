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
    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
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


int main(int argc, char** argv) {
    if (argc != 3) {
	return 1;
    }
    
    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);
    
    SDL_Init(SDL_INIT_VIDEO);
    srand(time(NULL));

    SDL_Window* window = SDL_CreateWindow("Pathtracer",
					  SDL_WINDOWPOS_CENTERED,
					  SDL_WINDOWPOS_CENTERED,
					  width, height,
					  0);
    SDL_Surface* screen = SDL_GetWindowSurface(window);
    wipe_surface(screen);

    bool running = true;

    uint8_t* cells[2];
    cells[0]= malloc(width * height * 2);
    cells[1] = cells[0] + width * height;

    for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	    cells[0][y * width + x] = (rand() % 100) <= 5;
	}
    }

    copy_redundant_borders(cells[0], width, height);

    int i = 0;
    while (running) {
	SDL_Event evt;
	while (SDL_PollEvent(&evt)) {
	    if (evt.type == SDL_QUIT) {
		running = false;
	    }
	}

	update_cells(cells[i%2], cells[(i+1)%2], width, height);
	render_cells(screen, cells[(i+1)%2], width, height);
	SDL_UpdateWindowSurface(window);

	i++;
    }

    free(cells[0]);
    SDL_Quit();
    return 0;
}
