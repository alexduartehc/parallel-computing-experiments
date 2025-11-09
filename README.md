# parallel-computing-experiments

This repository showcases two small, focused HPC/parallel programming projects:
  * Threaded Matrix Multiplication (Pthreads, C): compares single-threaded vs. multi-threaded performance and reports speedup.
  * Distributed Numerical Integration (MPI, C++): evaluates definite integrals with Simpson’s rule across ranks and reports timing.


## Project 1: Threaded Matrix Multiplication (Pthreads)

What it does:
  * Multiplies dense matrices A (r×c) and B (c×k) to produce C (r×k).
  * Implements both single-thread and multi-thread kernels and times each across multiple iterations, reporting speedup and % improvement.
  * Supports file-based or random matrix generation (ints/floats/mixed) and can save a full report (inputs, outputs, timings). 

Key implementation notes:
  * Internal contiguous buffers + row pointers for cache-friendly access.
  * Work is partitioned by row ranges per thread; each thread computes a block of output rows.
  * Timing via clock_gettime(CLOCK_MONOTONIC); average over user-specified iterations.


## Project 2: Numerical Integration with MPI (Simpson’s Rule)

What it does:
  * Uses composite Simpson’s rule distributed across p MPI ranks to approximate
  
![Integral](https://latex.codecogs.com/svg.image?\int_a^b%20f(x)\,dx)​

Each rank integrates its local subinterval; rank 0 reduces & prints the result + timing.
  * Integrands selectable via a flag: (default: cos)

    0: sin(x); 1: cos(x); 2: tan(x); 3: 1/x 


## Learning Goals:
  * Practice data parallelism with Pthreads (row-wise work partitioning).
  * Practice distributed memory parallelism with MPI (domain decomposition + reduction).
  * Measure and interpret speedup and scaling behavior.


-----------


parallel-computing-experiments

│

├── matrix-multiplication-threads.c        # Pthreads matrix multiply with timing & report output

├── numerical-integration-mpi.cpp          # MPI Simpson’s rule integration with timing

├── README.md

└── LICENSE (optional)

