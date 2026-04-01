#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#include "allreduce.h" 

void print_theoretical_memory_analysis(int rank, int size, int count, 
                                      int segment_size, MPI_Datatype datatype); 

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the data size array to be tested
    int test_sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int ite_per_size = 10;
    
    // Traverse all test scales
    setlocale(LC_NUMERIC, "");
    for (int test_idx = 0; test_idx < num_tests; test_idx++) {
        int test_size = test_sizes[test_idx];
        
        if (rank == 0) {
            printf("\n=== Testing with data size: %'d , process number: %d ===\n", test_size, size);
        }
        
        float *send_data = (float*)malloc(test_size * sizeof(float));
        float *recv_data1 = (float*)malloc(test_size * sizeof(float));
        float *recv_data2 = (float*)malloc(test_size * sizeof(float));
        
        if (send_data == NULL || recv_data1 == NULL || recv_data2 == NULL) {
            fprintf(stderr, "Rank %d: Memory allocation failed for test_size = %d\n", 
                    rank, test_size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Initialize data
        for (int i = 0; i < test_size; i++) {
            send_data[i] = (float)(rank + 1);  // The data of each process is different
        }
        
        // Test Approach 1 : Reduce First, then Broadcast
        double total_time1 = 0.0;
        for(int i = 0; i < ite_per_size; i ++) {
            float *send_data_copy = (float*)malloc(test_size * sizeof(float));
            memcpy(send_data_copy, send_data, test_size * sizeof(float));
            MPI_Barrier(MPI_COMM_WORLD);

            double start_time1 = MPI_Wtime();
            allreduce_reduce_bcast(send_data_copy, recv_data1, test_size,
                               MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            double end_time1 = MPI_Wtime();
            
            total_time1 += (end_time1 - start_time1);
            free(send_data_copy);
        }
        double avg_time1 = total_time1 / ite_per_size;
        
        // Test Approach 2 : Ring Pipeline
        double total_time2 = 0.0;
        int segment_size;
        for(int i = 0; i < ite_per_size; i ++) {
            float *send_data_copy = (float*)malloc(test_size * sizeof(float));
            memcpy(send_data_copy, send_data, test_size * sizeof(float));
            MPI_Barrier(MPI_COMM_WORLD);

            double start_time2 = MPI_Wtime();
            all_reduce_ring_pipeline(rank, size, send_data_copy, recv_data2, test_size, 
                                    MPI_FLOAT, MPI_COMM_WORLD, &segment_size);
            double end_time2 = MPI_Wtime();

            total_time2 += (end_time2 - start_time2);
            free(send_data_copy);
        }
        double avg_time2 = total_time2 / ite_per_size;

        // Test Approach 3 : MPI_allreduce
        double total_time3 = 0.0;
        for(int i = 0; i < ite_per_size; i ++) {
            float *send_data_copy = (float*)malloc(test_size * sizeof(float));
            memcpy(send_data_copy, send_data, test_size * sizeof(float));
            MPI_Barrier(MPI_COMM_WORLD);
            float *recv_data3 = (float*)malloc(test_size * sizeof(float));

            double start_time3 = MPI_Wtime();
            MPI_Allreduce(send_data_copy, recv_data3, test_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            double end_time3 = MPI_Wtime();

            total_time3 += (end_time3 - start_time3);
            free(send_data_copy);
            free(recv_data3);
        }
        double avg_time3 = total_time3 / ite_per_size;

        // Validate result
        float expected_sum = 0;
        for (int i = 0; i < size; i++) {
            expected_sum += (i + 1);
        }
    
        int error1 = 0, error2 = 0;
        for (int i = 0; i < test_size; i++) {
            if (fabs(recv_data1[i] - expected_sum) > 1e-6) {
                error1 = 1;
                break;
            }
            if (fabs(recv_data2[i] - expected_sum) > 1e-6) {
                error2 = 1;
                break;
            }
        }
        
        // Print the result
        if (rank == 0) {
            printf("Test size                         : %'d\n", test_size);
            printf("Result                            : %s\n",error1 || error2 ? "ERROR" : "OK");
            printf("Reduce+Broadcast: Time            = %.6f s\n", 
                   avg_time1);
            printf("Ring Pipeline: Time               = %.6f s\n", 
                   avg_time2);
            printf("MPI_allreduce: Time               = %.6f s\n", 
                   avg_time3);
            printf("Speedup (Ring vs Reduce+Broadcast): %.2f\n", 
                    avg_time1 / avg_time2);
            
            
            // // Print memory analysis
            // print_theoretical_memory_analysis(rank, size, test_size, 
            //                                       segment_size, MPI_FLOAT);
        }
        
        free(send_data);
        free(recv_data1);
        free(recv_data2);
        
        // Wait for all processes to complete the current test before starting the next one
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

// Theoretical memory usage of computational algorithms
void print_theoretical_memory_analysis(int rank, int size, int count, 
                                      int segment_size, MPI_Datatype datatype) {
    if (rank != 0) return;
    
    int type_size;
    MPI_Type_size(datatype, &type_size);
    
    printf("\n=== Theoretical Memory Analysis ===\n");
    setlocale(LC_NUMERIC, "");
    printf("\ntest size = %'d\n", count);
    setlocale(LC_NUMERIC, "C");
    printf("\nprocess number = %d\n", size);
    
    // 1. Reduce+Bcast
    printf("\n1. Reduce + Broadcast:\n");
    printf("   Input buffer:      %10.2f MB\n", count * type_size / (1024.0 * 1024.0));
    printf("   Output buffer:     %10.2f MB\n", count * type_size / (1024.0 * 1024.0));
    printf("   MPI internal:      %10.2f MB (estimated)\n", 0.1);  // 大约100KB
    printf("   Total per process: %10.2f MB\n", 
           (2 * count * type_size) / (1024.0 * 1024.0) + 0.1);
    
    // 2. Ring Pipeline
    printf("\n2. Ring Pipeline:\n");
    printf("   Input buffer:      %10.2f MB\n", count * type_size / (1024.0 * 1024.0));
    printf("   Output buffer:     %10.2f MB\n", count * type_size / (1024.0 * 1024.0));
    
    int total_segments = (count + segment_size - 1) / segment_size;
    printf("   Segment size:           %d elements\n", segment_size);
    printf("   Total segments:           %d\n", total_segments);
    printf("   Send buffer:       %10.2f MB\n", 
           segment_size * type_size / (1024.0 * 1024.0));
    printf("   Recv buffer:       %10.2f MB\n", 
           segment_size * type_size / (1024.0 * 1024.0));
    printf("   Total buffers:     %10.2f MB\n", 
           (2 * segment_size * type_size) / (1024.0 * 1024.0));
    printf("   Total per process: %10.2f MB\n", 
           (2 * count * type_size + 2 * segment_size * type_size) / (1024.0 * 1024.0));
    
    // Comparison
    printf("\n3. Comparison:\n");
    double reduce_bcast_mem = (2 * count * type_size) / (1024.0 * 1024.0) + 0.1;
    double ring_mem = (2 * count * type_size + 2 * segment_size * type_size) / (1024.0 * 1024.0);
    double difference = ring_mem - reduce_bcast_mem;
    double percentage = (difference / reduce_bcast_mem) * 100.0;
    
    printf("   Reduce+Bcast:      %10.2f MB\n", reduce_bcast_mem);
    printf("   Ring Pipeline:     %10.2f MB\n", ring_mem);
    printf("   Difference:        %10.2f MB (%+.1f%%)\n", difference, percentage);
    
    if (difference > 0) {
        printf("   Ring uses MORE memory by %.1f%%\n", percentage);
    } else {
        printf("   Ring uses LESS memory by %.1f%%\n", -percentage);
    }

    printf("\n===================================\n");

}