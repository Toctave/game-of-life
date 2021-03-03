#include <stdint.h>

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
