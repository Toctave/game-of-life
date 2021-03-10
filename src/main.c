#include <stdio.h>
#include <assert.h>

#include "io.h"
#include "tile_indexing.h"
#include "geometry.h"

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

typedef struct {
    Vec2i pos;
    uint8_t* cells[2];               // device-side buffers
    uint8_t* margin_buffers_send[8]; // host-side buffers
    uint8_t* margin_buffers_recv[8]; // host-side buffers
    MPI_Request send_requests[8];
} Tile;

MPI_File g_log_file;
MPI_Comm g_worker_comm;

#define SAFE_CUDA(call) do {						\
	cudaError_t val = (call);					\
	if (val != cudaSuccess) {					\
	    fprintf(stderr, "Fatal error in call to" #call "\n");	\
	    exit(1);							\
	}								\
    } while (0)

#define WORKER_LOG(...) do {                                            \
        char buffer[512];                                               \
        sprintf(buffer, __VA_ARGS__);                                   \
        MPI_File_write_ordered(g_log_file, buffer, strlen(buffer), MPI_BYTE, NULL); \
    } while (0)

void* safe_cuda_malloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    
    if (err != cudaSuccess) {
	fprintf(stderr, "Fatal error in cuda malloc, exiting.\n");
	exit(1);
    }

    return ptr;
}

void safe_cuda_free(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    
    if (err != cudaSuccess) {
	fprintf(stderr, "Fatal error in cuda free, exiting.\n");
	exit(1);
    }
}

void safe_cuda_memcpy(void* dst, const void* src, size_t size, enum cudaMemcpyKind kind) {
    cudaError_t err = cudaMemcpy(dst, src, size, kind);

    if (err != cudaSuccess) {
	fprintf(stderr, "Fatal error in cuda memcpy, exiting.\n");
	exit(1);
    }
}

static uint8_t rule(uint8_t alive, uint8_t neighbours) {
    return (neighbours == 3) || (alive && (neighbours == 2));
}

static void send_to_neighbour(uint8_t* buffer,
                              size_t size,
                              Vec2i dst_pos,
                              int my_index,
                              const GridDimensions* gd,
                              MPI_Request* req) {
    int neighbour = rank_of_tile(dst_pos, gd);
    int tag = (my_index << 16) | index_of_tile(dst_pos, gd);
    MPI_Isend(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD, req);
}

static void recv_from_neighbour(uint8_t* buffer,
                                size_t size,
                                Vec2i src_pos,
                                int my_index,
                                const GridDimensions* gd) {
    int neighbour = rank_of_tile(src_pos, gd);
    int tag = (index_of_tile(src_pos, gd) << 16) | my_index;

    // printf("receiving data from tile (%d, %d)\n", tag, srcx, srcy);
    MPI_Recv(buffer, size, MPI_BYTE, neighbour, tag, MPI_COMM_WORLD, NULL);
}

static void update_tile_inside(uint8_t* src, uint8_t* dst, int wide_size, int margin_width) {
    int neighbour_offsets[8] = {
        -wide_size - 1, -wide_size, -wide_size + 1,
        -1,                                      1,
        wide_size - 1,   wide_size,  wide_size + 1
    };

    int start = margin_width;
    int end = wide_size - margin_width;

    /* Update internal tile */
    for (int j = start; j < end; j++) {
        for (int i = start; i < end; i++) {
            int neighbours = 0;
            int base = j * wide_size + i;
            for (int k = 0; k < 8; k++) {
                neighbours += src[base + neighbour_offsets[k]];
            }
            dst[base] = rule(src[base], neighbours);
        }
    }
}

/* 
 * - cells : device pointer
 * - buffer : host pointer
 */
static void pack_sub_grid(const uint8_t* cells, int wide_size, Rect rect, uint8_t* buffer) {
    int row_length = (rect.max.x - rect.min.x);
    for (int y = rect.min.y; y < rect.max.y; y++) {
        int cells_offset = y * wide_size + rect.min.x;
        int buffer_offset = (y - rect.min.y) * row_length;
        
        safe_cuda_memcpy(buffer + buffer_offset,
			 cells + cells_offset,
			 row_length,
			 cudaMemcpyDeviceToHost);
    }
}

/* 
 * - cells : device pointer
 * - buffer : host pointer
 */
static void unpack_sub_grid(uint8_t* cells, int wide_size, Rect rect, const uint8_t* buffer) {
    int row_length = (rect.max.x - rect.min.x);
    for (int y = rect.min.y; y < rect.max.y; y++) {
        int cells_offset = y * wide_size + rect.min.x;
        int buffer_offset = (y - rect.min.y) * row_length;
        
        safe_cuda_memcpy(cells + cells_offset,
			 buffer + buffer_offset,
			 row_length,
			 cudaMemcpyHostToDevice);
    }
}

static void send_margins(Tile* tile, int src_index, const GridDimensions* gd) {
    assert(src_index == 0 || src_index == 1);
    uint8_t* cells = tile->cells[src_index];
    
    int my_index = index_of_tile(tile->pos, gd);

    for (NeighbourIndex i = 0; i < NEIGHBOUR_INDEX_COUNT; i++) {
        Vec2i nb = neighbour_tile(tile->pos, i, gd);        
        pack_sub_grid(cells,
                      gd->wide_size,
                      gd->subregions_send[i],
                      tile->margin_buffers_send[i]);
        send_to_neighbour(tile->margin_buffers_send[i],	
                          gd->subregion_sizes[i],			
                          nb,
                          my_index,			
                          gd,
                          &tile->send_requests[i]);
    }
}

static void wait_for_send_completion(Tile* tile) {
    for (int i = 0; i < 8; i++) {
        MPI_Wait(&tile->send_requests[i], NULL);
    }
}

static void recv_margins(Tile* tile, int src_index, const GridDimensions* gd) {
    uint8_t* cells = tile->cells[src_index];
    
    int my_index = index_of_tile(tile->pos, gd);

    for (NeighbourIndex i = 0; i < NEIGHBOUR_INDEX_COUNT; i++) {
        Vec2i nb = neighbour_tile(tile->pos, i, gd);
        recv_from_neighbour(tile->margin_buffers_recv[i],
                            gd->subregion_sizes[i],
                            nb,
                            my_index,			
                            gd);
        unpack_sub_grid(cells,
                        gd->wide_size,
                        gd->subregions_recv[i],
                        tile->margin_buffers_recv[i]);
    }
}

void update_tile_kernel_call(uint8_t* src,
			     uint8_t* dst,
			     int wide_size,
			     int margin_width);

static void worker(int rank, int iter, const GridDimensions* gd) {
    double tstart = MPI_Wtime();
    
    const Rect grid_center = {
        gd->margin_width,
        gd->margin_width,
        gd->tile_size + gd->margin_width,
        gd->tile_size + gd->margin_width
    };
    
    int tile_count;
    MPI_Recv(&tile_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    Tile* tiles = malloc(sizeof(Tile) * tile_count);

    size_t cell_count = gd->wide_size * gd->wide_size;
    size_t total_margin_area = 4 * gd->margin_width * (gd->margin_width + gd->tile_size);

    size_t cell_buffer_size = cell_count * 2 * tile_count;
    uint8_t* cell_buffer = safe_cuda_malloc(sizeof(uint8_t) * cell_buffer_size);
    
    for (int i = 0; i < tile_count; i++) {
        size_t tile_offset = cell_count * 2 * i; 
        tiles[i].cells[0] = &cell_buffer[tile_offset];
        tiles[i].cells[1] = &cell_buffer[tile_offset + cell_count];
    }

    size_t margin_buffer_size = total_margin_area * 2 * tile_count;
    uint8_t* margin_buffer = malloc(sizeof(uint8_t) * margin_buffer_size);
    
    for (int i = 0; i < tile_count; i++) {
	size_t tile_offset = total_margin_area * 2 * i;
        size_t margin_offset = 0;
	
        for (int j = 0; j < 8; j++) {
            tiles[i].margin_buffers_send[j] =
                &margin_buffer[tile_offset + margin_offset];
            margin_offset += gd->subregion_sizes[j];

            tiles[i].margin_buffers_recv[j] =
                &margin_buffer[tile_offset + margin_offset];
            margin_offset += gd->subregion_sizes[j]; 
        }
	assert(margin_offset == total_margin_area * 2);
    }

    uint8_t* recv_buffer = malloc(sizeof(uint8_t) * gd->tile_size * gd->tile_size);
    
    for (int i = 0; i < tile_count; i++) {
        MPI_Status status;
        MPI_Recv(recv_buffer, gd->tile_size*gd->tile_size, MPI_BYTE,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
        
        unpack_sub_grid(tiles[i].cells[0],
                        gd->wide_size,
                        grid_center,
                        recv_buffer);

        tiles[i].pos = tile_of_index(status.MPI_TAG, gd);
    }

    free(recv_buffer);
    
    double tinit = MPI_Wtime();
    int src_index = 0;

    double tsend = 0.0, trecv = 0.0, tupdate = 0.0, twaitsend=0.0;

    int current_step = 0;
    while (current_step < iter) {
        double t0 = MPI_Wtime();
        
        for (int ti = 0; ti < tile_count; ti++) {
            send_margins(&tiles[ti],
                         src_index,
                         gd);
        }
            
        double t1 = MPI_Wtime();
	
        for (int ti = 0; ti < tile_count; ti++) {
            recv_margins(&tiles[ti],
                         src_index,
                         gd);
        }

        double t2 = MPI_Wtime();
	
        for (int growing_margin = 1;
             growing_margin <= gd->margin_width;
             growing_margin++) {
             #pragma omp parallel
	    {
                #pragma omp for schedule(dynamic)
		for (int ti = 0; ti < tile_count; ti++) {
		    update_tile_kernel_call(tiles[ti].cells[src_index],
					    tiles[ti].cells[!src_index],
					    gd->wide_size,
					    growing_margin);
		}
	    }
            src_index = !src_index;
            current_step++;
            if (current_step >= iter) {
                break;
            }
        }
	
        double t3 = MPI_Wtime();
	
        for (int ti = 0; ti < tile_count; ti++) {
            wait_for_send_completion(&tiles[ti]);
        }

        double t4 = MPI_Wtime();

        tsend += t1 - t0;
        trecv += t2 - t1;
        tupdate += t3 - t2;
        twaitsend += t4 - t3;
    }

    double tfinish_iter = MPI_Wtime();
    
    uint8_t* send_buffer = malloc(sizeof(uint8_t) * gd->tile_size * gd->tile_size);

    for (int ti = 0; ti < tile_count; ti++) {
        pack_sub_grid(tiles[ti].cells[iter%2],
                      gd->wide_size,
                      grid_center,
                      send_buffer);
        int tag = index_of_tile(tiles[ti].pos, gd);
        MPI_Send(send_buffer, gd->tile_size * gd->tile_size, MPI_BYTE, 0, tag, MPI_COMM_WORLD);
    }

    double tfinish_send = MPI_Wtime();

    free(send_buffer);
    safe_cuda_free(cell_buffer);
    free(tiles);

    WORKER_LOG("%d, %g, %g, %g, %g, %g, %g, %g\n",
	       rank,
	       tfinish_send - tstart,
	       tinit - tstart,
	       tsend,
	       trecv,
	       tupdate,
	       twaitsend,
	       tfinish_send - tfinish_iter);
}

void init_tiles_randomly(uint8_t *tiles, int total_cell_count, float density){
    for (int i = 0; i < total_cell_count; i++){
        tiles[i] = ((float) rand()) / RAND_MAX <= density;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, node_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &node_count);

    MPI_Comm_split(MPI_COMM_WORLD,
                   rank == 0 ? 0 : 1,
                   0,
                   &g_worker_comm);
    
    /* Initialize the cells */
    Options options;
    if (!parse_options(&options, argc, argv)) {
        fprintf(stderr,
                "Usage :\n"
                "  game_of_life [-w <width>] [-h <height>] [-d <density>] [-t <tile size>] [-i <iter>] [-g] [-s <seed>] [-o <output_file>] [-f <input_filepath>]\n");
        return 1;
    }

    SAFE_CUDA(cudaGetDeviceCount(&options.gpu_count));

    printf("Device count : %d\n", options.gpu_count);
    
    srand(options.seed);

    /* Init tiles and scatter them to worker nodes */
    assert(options.width % options.tile_size == 0);
    assert(options.height % options.tile_size == 0);

    int mw = options.margin_width;
    int ts = options.tile_size;
    int corner_area = mw * mw;
    int border_area = ts * mw;
    
    GridDimensions gd = {
        .width = options.width,
        .height = options.height,
        .tile_hcount = options.width / options.tile_size,
        .tile_vcount = options.height / options.tile_size,
        .tile_size = options.tile_size,
        .wide_size = options.tile_size + 2 * options.margin_width,
        .margin_width = options.margin_width,

        .node_count = node_count,
        .subregions_send = {
            [TOPLEFT] = { mw, mw, 2 * mw, 2 * mw },
            [TOP] = { mw, mw, mw + ts, 2 * mw },
            [TOPRIGHT] = { ts, mw, ts + mw, 2 * mw },
            [LEFT] = { mw, mw, 2 * mw, mw + ts},
            [RIGHT] = { ts, mw, ts + mw, ts + mw},
            [BOTLEFT] = { mw, ts, 2 * mw, ts + mw},
            [BOT] = { mw, ts, ts + mw, ts + mw },
            [BOTRIGHT] = { ts, ts, ts + mw, ts + mw },
        },
        .subregions_recv = {
            [TOPLEFT] = { 0, 0, mw, mw },
            [TOP] = { mw, 0, mw + ts, mw },
            [TOPRIGHT] = { ts + mw, 0, ts + 2 * mw, mw },
            [LEFT] = { 0, mw, mw, mw + ts},
            [RIGHT] = { ts + mw, mw, ts + 2 * mw, ts + mw},
            [BOTLEFT] = { 0, ts + mw, mw, ts + 2 * mw},
            [BOT] = { mw, ts + mw, ts + mw, ts + 2 * mw },
            [BOTRIGHT] = { ts + mw, ts + mw, ts + 2 * mw, ts + 2 * mw },
        },
        .subregion_sizes = {
            [TOPLEFT] = corner_area,
            [TOP] = border_area,
            [TOPRIGHT] = corner_area,
            [LEFT] = border_area,
            [RIGHT] = border_area,
            [BOTLEFT] = corner_area,
            [BOT] = border_area,
            [BOTRIGHT] = corner_area,
        }
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
	

        uint8_t *tmp_tile = malloc(sizeof(uint8_t) * options.tile_size*options.tile_size);
        for (int i = 0; i < gd.tile_vcount*gd.tile_hcount; i++) {
            MPI_Status status;
            MPI_Recv(tmp_tile, options.tile_size*options.tile_size, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            memcpy(&tiles[options.tile_size * options.tile_size * status.MPI_TAG],
		   tmp_tile,
		   options.tile_size * options.tile_size * sizeof(uint8_t));
        }

        if (options.output_filepath) {
            save_grid_to_png(tiles, options.output_filepath, &gd);
        } else {
            print_grid(tiles, &gd);
        }

        free(tmp_tile);
        free(tiles);
    } else{
        MPI_File_open(g_worker_comm,
                      options.log_filepath,
                      MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_SEQUENTIAL,
                      MPI_INFO_NULL,
                      &g_log_file);

        worker(rank, options.iter, &gd);
    }

    MPI_File_close(&g_log_file);

    MPI_Finalize();
}
