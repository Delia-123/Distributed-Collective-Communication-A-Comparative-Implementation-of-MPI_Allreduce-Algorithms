#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Implementation of Allreduce based on Reduce + Roadcast (Naive)
 * 
 * @param sendbuf   Send buffer (input)
 * @param recvbuf   Receive buffer (output)
 * @param count     Number of data elements
 * @param datatype  Datatype of data
 * @param op        Operation
 * @param comm      Communicator
 */
void allreduce_reduce_bcast(void *sendbuf, void *recvbuf, int count,
                           MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    int root = 0;  // Fixed selection of process 0 as root
    
    // Step 1: All processes use MPI_Reduce to reduce data to the root process
    MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    
    // Step 2: The root process uses MPI_Bcast to broadcast the results to all processes
    MPI_Bcast(recvbuf, count, datatype, root, comm);
}