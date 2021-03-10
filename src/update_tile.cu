#include <stdint.h>
#include <stdio.h>

__device__
uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

__global__
void update_tile_inside_gpu(uint8_t* src, uint8_t* dst, int wide_size, int margin_width) {
    int neighbour_offsets[8] = {
        -wide_size - 1, -wide_size, -wide_size + 1,
        -1,                                      1,
        wide_size - 1,   wide_size,  wide_size + 1
    };    

    int start = margin_width;
    int end = wide_size - margin_width;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    /* printf("i = %d, j = %d\n", i, j); */
    if (i >= start && i < end
	&& j >= start && j < end) {

	int neighbours = 0;
	int base = j * wide_size + i;
	for (int k = 0; k < 8; k++) {
	    neighbours += src[base + neighbour_offsets[k]];
	}
	dst[base] = rule(src[base], neighbours);
    }
}

extern "C" 
__host__
void update_tile_kernel_call(uint8_t* src,
			     uint8_t* dst,
			     int wide_size,
			     int margin_iterations) {
    dim3 numBlocks(wide_size / 32 + 1, wide_size / 32 + 1);
    dim3 threadsPerBlocks(32, 32);
    for (int growing_margin = 1;
	 growing_margin <= margin_iterations;
	 growing_margin++) {
	update_tile_inside_gpu<<< numBlocks, threadsPerBlocks >>>(src, dst, wide_size, growing_margin);
    }
}
