#!/usr/bin/env python3
"""
Generate large, computationally challenging test files for energy_storms.

File format:
  Line 1: number of particles (N)
  Lines 2..N+1: <position> <energy>
    position: integer in [0, array_size - 1]
    energy:   integer in [1, 1000000]
"""

import random
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_wave(filename, array_size, num_particles, seed=None):
    """Generate a single wave file."""
    if seed is not None:
        random.seed(seed)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(f"{num_particles}\n")
        for _ in range(num_particles):
            position = random.randint(0, array_size - 1)
            energy = random.randint(1, 1_000_000)
            f.write(f"{position} {energy}\n")
    size_kb = os.path.getsize(path) // 1024
    print(f"  Generated: {filename}  ({num_particles:,} particles, {size_kb} KB)")


def generate_test(name, array_size, num_particles, num_waves, base_seed=42):
    print(f"\n[{name}] array={array_size:,}  particles={num_particles:,}  waves={num_waves}")
    for w in range(1, num_waves + 1):
        fname = f"{name}_w{w}"
        generate_wave(fname, array_size, num_particles, seed=base_seed + w)


# -----------------------------------------------------------------------
# test_10: Massive array (500M positions), sparse particles
#          stresses reduction and boundary propagation across huge memory.
# -----------------------------------------------------------------------
generate_test(
    name="test_10_a500M_p100",
    array_size=500_000_000,
    num_particles=100,
    num_waves=4,
    base_seed=100,
)

# -----------------------------------------------------------------------
# test_11: Medium array (10M positions), extremely dense particles
#          stresses the propagation loop with maximum memory bandwidth.
# -----------------------------------------------------------------------
generate_test(
    name="test_11_a10M_p500k",
    array_size=10_000_000,
    num_particles=500_000,
    num_waves=6,
    base_seed=200,
)

# -----------------------------------------------------------------------
# test_12: Large array (50M positions), high particle count (100k)
#          balanced workload for comparing seq vs parallel speedup.
# -----------------------------------------------------------------------
generate_test(
    name="test_12_a50M_p100k",
    array_size=50_000_000,
    num_particles=100_000,
    num_waves=8,
    base_seed=300,
)

# -----------------------------------------------------------------------
# test_13: Worst-case reduction: 200M positions, all particles at same
#          position (maximum collision / reduction contention).
# -----------------------------------------------------------------------
print(f"\n[test_13] array=200,000,000  particles=1,000,000  waves=4  (hotspot stress test)")
for w in range(1, 5):
    fname = f"test_13_a200M_p1M_hotspot_w{w}"
    path = os.path.join(OUTPUT_DIR, fname)
    random.seed(400 + w)
    hotspot = random.randint(0, 199_999_999)
    with open(path, "w") as f:
        f.write(f"1000000\n")
        for _ in range(1_000_000):
            energy = random.randint(1, 1_000_000)
            f.write(f"{hotspot} {energy}\n")
    size_kb = os.path.getsize(path) // 1024
    print(f"  Generated: {fname}  (hotspot={hotspot:,}, {size_kb} KB)")

# -----------------------------------------------------------------------
# test_14: GPU stress test: 2M positions, 1M particles, 10 waves.
#          Dense random distribution for CUDA thread divergence stress.
# -----------------------------------------------------------------------
generate_test(
    name="test_14_a2M_p1M",
    array_size=2_000_000,
    num_particles=1_000_000,
    num_waves=10,
    base_seed=500,
)

# -----------------------------------------------------------------------
# test_15: MPI boundary stress: tiny array (1000 positions), max
#          particles. Forces heavy inter-process communication.
# -----------------------------------------------------------------------
generate_test(
    name="test_15_a1k_p500k",
    array_size=1_000,
    num_particles=500_000,
    num_waves=6,
    base_seed=600,
)

print("\nDone! All test files generated.")
print("\nRun commands:")
print("  ./energy_storms_seq 10000000  test_files/test_11_a10M_p500k_w1 test_files/test_11_a10M_p500k_w2 test_files/test_11_a10M_p500k_w3")
print("  ./energy_storms_seq 50000000  test_files/test_12_a50M_p100k_w1 test_files/test_12_a50Mi_p100k_w2")
print("  ./energy_storms_seq 2000000   test_files/test_14_a2M_p1M_w1 test_files/test_14_a2M_p1M_w2")
