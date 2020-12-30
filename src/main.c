#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#include <mmintrin.h>

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

static void print_m128i(__m128i a) {
    uint8_t* p = (uint8_t*) &a;
    printf("[ ");
    for (int i = 0; i < 16; i++) {
	printf("%d ", p[i]);
    }
    printf("]\n");
}

static void update_cells_sse(GolData* gol) {
    assert(width % 16 == 0);

    uint8_t* src = gol->cells[gol->iterations % 2];
    uint8_t* dst = gol->cells[(gol->iterations + 1) % 2];

    __m128i shift_left = _mm_setr_epi8(1, 2, 3, 4,
				      5, 6, 7, 8,
				      9, 10, 11, 12,
				      13, 14, 15, -1);
    __m128i shift_right = _mm_setr_epi8(-1, 0, 1, 2,
				       3, 4, 5, 6,
				       7, 8, 9, 10,
				       11, 12, 13, 14);
    __m128i select_left = _mm_setr_epi8(-1, -1, -1, -1,
				       -1, -1, -1, -1,
				       -1, -1, -1, -1,
				       -1, -1, -1, 0);
    __m128i select_right = _mm_setr_epi8(15, -1, -1, -1,
					-1, -1, -1, -1,
					-1, -1, -1, -1,
					-1, -1, -1, -1);

    for (int j = gol->marginy; j < gol->height - gol->marginy; j++) {
	for (int i = gol->marginx; i < gol->width - gol->marginx; i += 16) {
	    int base = j * gol->width + i;
	    __m128i* a = (__m128i*) &src[base];
	    __m128i neighbours = _mm_setzero_si128();

	    /* Right and left */
	    __m128i ar = _mm_shuffle_epi8(*a, shift_right);
	    __m128i al = _mm_shuffle_epi8(*a, shift_left);

	    neighbours = _mm_adds_epu8(ar, al);

	    /* Top */
	    __m128i* top = (__m128i*) &src[base - gol->width];
	    __m128i topr = _mm_shuffle_epi8(*top, shift_right);
	    __m128i topl = _mm_shuffle_epi8(*top, shift_left);

	    neighbours = _mm_adds_epu8(neighbours, *top);
	    neighbours = _mm_adds_epu8(neighbours, topr);
	    neighbours = _mm_adds_epu8(neighbours, topl);

	    /* Bottom */
	    __m128i* bot = (__m128i*) &src[base + gol->width];
	    __m128i botr = _mm_shuffle_epi8(*bot, shift_right);
	    __m128i botl = _mm_shuffle_epi8(*bot, shift_left);

	    neighbours = _mm_adds_epu8(neighbours, *bot);
	    neighbours = _mm_adds_epu8(neighbours, botr);
	    neighbours = _mm_adds_epu8(neighbours, botl);	    

	    /* borders (top-left, left, bottom-left, top-right, right, bottom-right) */
	    __m128i* border = (__m128i*) &src[base - gol->width - 16];
	    __m128i border_shifted = _mm_shuffle_epi8(*border, select_right);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);

	    border = (__m128i*) &src[base - 16];
	    border_shifted = _mm_shuffle_epi8(*border, select_right);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);

	    border = (__m128i*) &src[base + gol->width - 16];
	    border_shifted = _mm_shuffle_epi8(*border, select_right);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);

	    border = (__m128i*) &src[base - gol->width + 16];
	    border_shifted = _mm_shuffle_epi8(*border, select_left);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);
	    
	    border = (__m128i*) &src[base + 16];
	    border_shifted = _mm_shuffle_epi8(*border, select_left);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);
	    
	    border = (__m128i*) &src[base + gol->width + 16];
	    border_shifted = _mm_shuffle_epi8(*border, select_left);
	    neighbours = _mm_adds_epu8(neighbours, border_shifted);

	    __m128i* dst128 = (__m128i*) &dst[base];
	    *dst128 = _mm_or_si128(
		_mm_and_si128(*a,
			      _mm_cmpeq_epi8(neighbours, _mm_set1_epi8(2))),
		_mm_cmpeq_epi8(neighbours, _mm_set1_epi8(3)));
	    *dst128 = _mm_and_si128(*dst128,
				    _mm_set1_epi8(1));
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
	update_cells_sse(gol);
	gol->iterations++;

	if (gol->stop)
	    return 0;
    }
    return 0;
}

static void initialize_gol(GolData* gol) {
    gol->cells[0] = _mm_malloc(gol->width * gol->height * 2, sizeof(__m128i));
    gol->cells[1] = gol->cells[0] + gol->width * gol->height;

    for (int y = 0; y < gol->height; y++) {
	for (int x = 0; x < gol->width; x++) {
	    gol->cells[0][y * gol->width + x] = 0;
		/* ((float) rand() / RAND_MAX) <= density; */
	}
    }

    gol->iterations = 0;
    gol->stop = false;
}

static void free_gol(GolData* gol) {
    free(gol->cells[0]);
}

static void read_rle(GolData* gol, const char* filename) {
    memset(gol->cells[0], 0, gol->width * gol->height);
    
    FILE* f = fopen(filename, "r");
    if (f == NULL) {
	return;
    }

    int m, n;
    if (fscanf(f, "x = %d, y = %d\n", &m, &n) < 2) {
	fclose(f);
	return;
    }

    if (m > gol->width || n > gol->height) {
	fclose(f);
	return;
    }

    int x = 0;
    int y = 0;
    while (true) {
	int run;
	char tag;
	if (fscanf(f, " %d%c", &run, &tag) < 2) {
	    if (fscanf(f, " %c", &tag) < 1) {
		fclose(f);
		return;
	    }
	    if (tag == '!')
		break;
	    run = 1;
	}

	if (tag == 'b')
	    x += run;
	if (tag == 'o') {
	    int idx = (y + gol->marginy) * gol->width
		+ (x + gol->marginx);
	    memset(&gol->cells[0][idx], 1, run);
	    x += run;
	}
	if (tag == '$') {
	    y += run;
	    x = 0;
	}
    }

    fclose(f);
    copy_redundant_borders(gol);
}

int main(int argc, char** argv) {
    if (argc != 4) {
	fprintf(stderr, "Usage :\n  game_of_life <width> <height> <initial proportion of live cells>\n");
	return 1;
    }
    
    srand(time(NULL));

    GolData gol;
    gol.marginx = 16;
    gol.marginy = 1;
    gol.width = atoi(argv[1]) + gol.marginx * 2;
    gol.height = atoi(argv[2]) + gol.marginy * 2;
    initialize_gol(&gol);

    read_rle(&gol, argv[3]);
    
    SDL_Init(SDL_INIT_VIDEO);
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
	    if (evt.type == SDL_QUIT) {
		gol.stop = true;
	    }
	    if (evt.type == SDL_KEYDOWN) {
		update_cells_sse(&gol);
		gol.iterations++;
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
