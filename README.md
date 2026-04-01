# Distributed-Collective-Communication-A-Comparative-Implementation-of-MPI_Allreduce-Algorithms
This project presents a comprehensive study implementing and comparing two core algorithms for the MPI_Allreduce operation: the Reduce-Broadcast method and the Ring Pipeline algorithm. The project moved beyond theoretical understanding to practical implementation and empirical evaluation on a multi-node cluster.



## Execution
<pre>
cd The/Folder/of/The/Project
module load mpi/openmpi/4.1.4-gcc-4.8.5
mpicc main.c allreduce_reduce_bcast.c all_reduce_ring_pipeline.c -o main -lm
mpirun -np 10 ./main
</pre>
