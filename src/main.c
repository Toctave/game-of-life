#include <stdio.h>
#include <stdbool.h>
#include <SDL2/SDL.h>

int main(int argc, char** argv) {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Pathtracer", 0, 0, 200, 200, 0);

    bool running = true;
    
    while (running) {
	SDL_Event evt;
	while (SDL_PollEvent(&evt)) {
	    if (evt.type == SDL_QUIT) {
		running = false;
	    }
	}
    }
    
    printf("Hello, world!\n");

    SDL_Quit();
    return 0;
}
