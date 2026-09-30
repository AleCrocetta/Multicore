#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "energy_storms.h"

#define BLOCK_SIZE 256

typedef struct {
    float value;
    int position;
} Maximum;

static void check_cuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ Maximum better_maximum(Maximum first, Maximum second) {
    if (second.value > first.value ||
        (second.value == first.value && second.position < first.position))
        return second;
    return first;
}

__global__ void accumulate_kernel(float *layer, int layer_size,
                                  const int *particles, int storm_size,
                                  float threshold) {
    extern __shared__ int shared_positions[];
    float *shared_energies = (float *)(shared_positions + blockDim.x);

    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    /* Inactive threads must still reach every block-wide barrier. */
    bool active = cell < layer_size;
    float value = active ? layer[cell] : 0.0f;

    for (int tile_start = 0; tile_start < storm_size;
         tile_start += blockDim.x) {
        int particle = tile_start + threadIdx.x;
        int tile_size = storm_size - tile_start;
        if (tile_size > blockDim.x) tile_size = blockDim.x;

        if (particle < storm_size) {
            shared_positions[threadIdx.x] = particles[particle * 2];
            shared_energies[threadIdx.x] =
                ((float)particles[particle * 2 + 1] * 1000.0f) / (float)layer_size;
        }
        __syncthreads();

        if (active) {
            for (int j = 0; j < tile_size; j++) {
                int distance = shared_positions[j] - cell;
                if (distance < 0) distance = -distance;

                float attenuation = sqrtf((float)(distance + 1));
                float contribution =
                    shared_energies[j] / attenuation;

                if (contribution >= threshold || contribution <= -threshold)
                    value += contribution;
            }
        }
        __syncthreads();
    }

    if (active)
        layer[cell] = value;
}

__global__ void relax_kernel(const float *source, float *destination,
                             int layer_size) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= layer_size) return;

    if (cell == 0 || cell == layer_size - 1)
        destination[cell] = source[cell];
    else
        destination[cell] =
            (source[cell - 1] + source[cell] + source[cell + 1]) / 3.0f;
}

__global__ void find_maxima_kernel(const float *layer, int layer_size,
                                   Maximum *block_maxima) {
    __shared__ Maximum shared_maxima[BLOCK_SIZE];

    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    Maximum candidate = { 0.0f, 0 };

    if (cell > 0 && cell < layer_size - 1) {
        float value = layer[cell];
        if (value > layer[cell - 1] && value > layer[cell + 1] && value > 0.0f) {
            candidate.value = value;
            candidate.position = cell;
        }
    }

    shared_maxima[threadIdx.x] = candidate;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            shared_maxima[threadIdx.x] =
                better_maximum(shared_maxima[threadIdx.x],
                               shared_maxima[threadIdx.x + offset]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        block_maxima[blockIdx.x] = shared_maxima[0];
}

__global__ void reduce_maxima_kernel(const Maximum *input, int count,
                                     Maximum *output) {
    __shared__ Maximum shared_maxima[BLOCK_SIZE];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    Maximum candidate = { 0.0f, 0 };
    if (index < count)
        candidate = input[index];

    shared_maxima[threadIdx.x] = candidate;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            shared_maxima[threadIdx.x] =
                better_maximum(shared_maxima[threadIdx.x],
                               shared_maxima[threadIdx.x + offset]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        output[blockIdx.x] = shared_maxima[0];
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum,
          int *positions) {
    if (layer_size <= 0 || num_storms <= 0)
        return;

    int layer_blocks = (layer_size - 1) / BLOCK_SIZE + 1;
    int max_particles = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_particles)
            max_particles = storms[i].size;

    float *device_layers[2] = { NULL, NULL };
    int *device_particles = NULL;
    Maximum *device_maxima[2] = { NULL, NULL };
    Maximum *device_results = NULL;

    check_cuda(cudaMalloc((void **)&device_layers[0],
                          sizeof(float) * (size_t)layer_size),
               "allocating the first layer");
    check_cuda(cudaMalloc((void **)&device_layers[1],
                          sizeof(float) * (size_t)layer_size),
               "allocating the second layer");
    if (max_particles > 0)
        check_cuda(cudaMalloc((void **)&device_particles,
                              sizeof(int) * (size_t)max_particles * 2),
                   "allocating the particle buffer");
    check_cuda(cudaMalloc((void **)&device_maxima[0],
                          sizeof(Maximum) * (size_t)layer_blocks),
               "allocating the first reduction buffer");
    check_cuda(cudaMalloc((void **)&device_maxima[1],
                          sizeof(Maximum) * (size_t)layer_blocks),
               "allocating the second reduction buffer");
    check_cuda(cudaMalloc((void **)&device_results,
                          sizeof(Maximum) * (size_t)num_storms),
               "allocating the result buffer");
    check_cuda(cudaMemset(device_layers[0], 0,
                          sizeof(float) * (size_t)layer_size),
               "initializing the layer");

    int current_layer = 0;
    size_t shared_bytes = (sizeof(int) + sizeof(float)) * BLOCK_SIZE;
    float threshold = THRESHOLD / (float)layer_size;

    for (int i = 0; i < num_storms; i++) {
        int storm_size = storms[i].size;

        if (storm_size > 0)
            check_cuda(cudaMemcpy(device_particles, storms[i].posval,
                                  sizeof(int) * (size_t)storm_size * 2,
                                  cudaMemcpyHostToDevice),
                       "copying storm particles");

        accumulate_kernel<<<layer_blocks, BLOCK_SIZE, shared_bytes>>>(
            device_layers[current_layer], layer_size, device_particles,
            storm_size, threshold);
        check_cuda(cudaGetLastError(), "launching the accumulation kernel");

        int next_layer = 1 - current_layer;
        relax_kernel<<<layer_blocks, BLOCK_SIZE>>>(
            device_layers[current_layer], device_layers[next_layer], layer_size);
        check_cuda(cudaGetLastError(), "launching the relaxation kernel");
        current_layer = next_layer;

        find_maxima_kernel<<<layer_blocks, BLOCK_SIZE>>>(
            device_layers[current_layer], layer_size, device_maxima[0]);
        check_cuda(cudaGetLastError(), "launching the local-maximum kernel");

        /* Keep reducing on the device until one value-position pair remains. */
        int candidate_count = layer_blocks;
        int current_maxima = 0;
        while (candidate_count > 1) {
            int reduction_blocks =
                (candidate_count - 1) / BLOCK_SIZE + 1;
            int next_maxima = 1 - current_maxima;

            reduce_maxima_kernel<<<reduction_blocks, BLOCK_SIZE>>>(
                device_maxima[current_maxima], candidate_count,
                device_maxima[next_maxima]);
            check_cuda(cudaGetLastError(), "launching a maximum reduction kernel");

            candidate_count = reduction_blocks;
            current_maxima = next_maxima;
        }

        check_cuda(cudaMemcpyAsync(&device_results[i],
                                   device_maxima[current_maxima],
                                   sizeof(Maximum), cudaMemcpyDeviceToDevice),
                   "storing a storm result");
    }

    Maximum *host_results =
        (Maximum *)malloc(sizeof(Maximum) * (size_t)num_storms);
    if (host_results == NULL) {
        fprintf(stderr, "Error: Allocating CUDA result memory\n");
        exit(EXIT_FAILURE);
    }

    check_cuda(cudaMemcpy(host_results, device_results,
                          sizeof(Maximum) * (size_t)num_storms,
                          cudaMemcpyDeviceToHost),
               "copying simulation results");

    for (int i = 0; i < num_storms; i++) {
        maximum[i] = host_results[i].value;
        positions[i] = host_results[i].position;
    }

    free(host_results);
    check_cuda(cudaFree(device_results), "freeing the result buffer");
    check_cuda(cudaFree(device_maxima[1]), "freeing the second reduction buffer");
    check_cuda(cudaFree(device_maxima[0]), "freeing the first reduction buffer");
    if (device_particles != NULL)
        check_cuda(cudaFree(device_particles), "freeing the particle buffer");
    check_cuda(cudaFree(device_layers[1]), "freeing the second layer");
    check_cuda(cudaFree(device_layers[0]), "freeing the first layer");
}
