#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <mpi.h>

void allreduce_reduce_bcast(void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

void all_reduce_ring_pipeline(int rank, int size, float *sendbuf, float *recvbuf,
                            int count, MPI_Datatype datatype, MPI_Comm comm, int *segment_size_ptr);

#endif