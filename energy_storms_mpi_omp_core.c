#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "energy_storms.h"

typedef struct {
    float value;
    int position;
} Maximum;

static void exchange_halos(float *layer, int local_size, int left, int right) {
    /* Exchange the first and last owned cells without ordering deadlocks. */
    MPI_Sendrecv(&layer[1], 1, MPI_FLOAT, left, 0,
                 &layer[local_size + 1], 1, MPI_FLOAT, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&layer[local_size], 1, MPI_FLOAT, right, 1,
                 &layer[0], 1, MPI_FLOAT, left, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    if (layer_size <= 0 || num_storms <= 0)
        return;

    int base_size = layer_size / num_processes;
    int remainder = layer_size % num_processes;
    int local_size = base_size + (rank < remainder);
    int global_start = rank * base_size + (rank < remainder ? rank : remainder);
    int left = local_size > 0 && global_start > 0 ? rank - 1 : MPI_PROC_NULL;
    int right = local_size > 0 && global_start + local_size < layer_size
                    ? rank + 1
                    : MPI_PROC_NULL;

    /* Owned cells use indices 1..local_size; indices 0 and local_size+1 are halos. */
    float *layer = (float *)calloc((size_t)local_size + 2, sizeof(float));
    float *layer_copy = (float *)calloc((size_t)local_size + 2, sizeof(float));

    int max_particles = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_particles)
            max_particles = storms[i].size;

    float *particle_energy = NULL;
    if (max_particles > 0)
        particle_energy = (float *)malloc(sizeof(float) * (size_t)max_particles);

    if (layer == NULL || layer_copy == NULL ||
        (max_particles > 0 && particle_energy == NULL)) {
        fprintf(stderr, "Error: Allocating MPI+OpenMP core memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    float threshold = THRESHOLD / (float)layer_size;

    for (int i = 0; i < num_storms; i++) {
        int storm_size = storms[i].size;
        int *storm_particles = storms[i].posval;

        for (int j = 0; j < storm_size; j++)
            particle_energy[j] =
                ((float)storm_particles[j * 2 + 1] * 1000.0f) / (float)layer_size;

        /* Each thread owns complete cells, so particle additions need no atomics. */
        #pragma omp parallel for schedule(static)
        for (int local_index = 1; local_index <= local_size; local_index++) {
            int global_index = global_start + local_index - 1;
            float value = layer[local_index];

            for (int j = 0; j < storm_size; j++) {
                int distance = storm_particles[j * 2] - global_index;
                if (distance < 0) distance = -distance;

                float attenuation = sqrtf((float)(distance + 1));
                float contribution = particle_energy[j] / attenuation;

                if (contribution >= threshold || contribution <= -threshold)
                    value += contribution;
            }

            layer[local_index] = value;
        }

        /* Accumulated neighbors are needed to relax partition-boundary cells. */
        exchange_halos(layer, local_size, left, right);

        #pragma omp parallel for schedule(static)
        for (int local_index = 1; local_index <= local_size; local_index++) {
            int global_index = global_start + local_index - 1;

            if (global_index == 0 || global_index == layer_size - 1)
                layer_copy[local_index] = layer[local_index];
            else
                layer_copy[local_index] =
                    (layer[local_index - 1] + layer[local_index] +
                     layer[local_index + 1]) / 3.0f;
        }

        float *swap = layer;
        layer = layer_copy;
        layer_copy = swap;

        /* Relaxed neighbors are needed to classify partition-boundary maxima. */
        exchange_halos(layer, local_size, left, right);

        Maximum local_maximum = { 0.0f, 0 };

        #pragma omp parallel
        {
            Maximum thread_maximum = { 0.0f, 0 };

            #pragma omp for schedule(static) nowait
            for (int local_index = 1; local_index <= local_size; local_index++) {
                int global_index = global_start + local_index - 1;
                float value = layer[local_index];

                if (global_index > 0 && global_index < layer_size - 1 &&
                    value > layer[local_index - 1] &&
                    value > layer[local_index + 1] &&
                    (value > thread_maximum.value ||
                     (value == thread_maximum.value &&
                      global_index < thread_maximum.position))) {
                    thread_maximum.value = value;
                    thread_maximum.position = global_index;
                }
            }

            #pragma omp critical
            {
                if (thread_maximum.value > local_maximum.value ||
                    (thread_maximum.value == local_maximum.value &&
                     thread_maximum.position < local_maximum.position))
                    local_maximum = thread_maximum;
            }
        }

        Maximum global_maximum = { 0.0f, 0 };
        MPI_Reduce(&local_maximum, &global_maximum, 1, MPI_FLOAT_INT,
                   MPI_MAXLOC, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            maximum[i] = global_maximum.value;
            positions[i] = global_maximum.position;
        }
    }

    free(particle_energy);
    free(layer_copy);
    free(layer);
}
