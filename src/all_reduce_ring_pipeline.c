#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>  
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <immintrin.h>

// Function declaration
int init_recv_seg(int rank, int size, int count, int past_seg_num, int segment_size, int total_segments,
                    bool isBcast, int *receive_seg_idx, int *receive_seg_start_idx, int *receive_seg_size);
int update_send_recv_seg(int size, int count, int past_seg_num, int segment_size, int total_segments,
                        int *send_seg_idx, int *start_idx, int *current_seg_size, 
                        int *receive_seg_idx, int *receive_seg_start_idx, int *receive_seg_size);

void all_reduce_ring_pipeline(int rank, int size, float *sendbuf, float *recvbuf, 
    int count, MPI_Datatype datatype, MPI_Comm comm, int *segment_size_ptr) {
    // Step 1: Initialize with local data
    memcpy(recvbuf, sendbuf, count * sizeof(float));
    // Only one process
    if(size < 2) {
        return;
    }
    
    // Step 2: Determine segment size for pipeline
    // Segment Size Choice 1:
    int segment_size;
    if (count <= 64 * 1024 / 8) {          // <= 64KB: 单段或小段
        segment_size = count;
    } else if (count <= 4 * 1024 * 1024 / 8) { // <= 4MB: 中等段
        segment_size = 64 * 1024 / 8;  // 64KB
    } else {                                // > 4MB: 大段
        segment_size = 256 * 1024 / 8; // 256KB
    }
    // Segment Size Choice 2:
    // int segment_size = 65536;

    if (segment_size > count) {
        segment_size = count;
    }
    if (segment_size_ptr != NULL) {
        *segment_size_ptr = segment_size;  
    }

    // Calculate the total number of segments (rounded up).
    // And the segmentation method is as follows: 
    //   the size of the preceding segments are all segment_size, 
    //   and the remaining segments are taken as the last segment.
    int total_segments = (segment_size + count - 1) / segment_size;
    
    // Step 3: Allocate buffers for pipeline
    float *send_seg = (float*)malloc(segment_size * sizeof(float));
    float *recv_seg = (float*)malloc(segment_size * sizeof(float));

    // Step 4: Ring neighbors
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    // Step 5: Pipeline the data collection and processing
    int send_seg_idx;
    int receive_seg_idx;
    int past_seg_num = 0;
    int start_idx;
    int current_seg_size;
    int receive_seg_start_idx;
    int receive_seg_size;
    // Do it for floor(total_segments/size) times, leaving the last iteration if it exists.
    for(int segments_of_size_loop_idx = 0; 
            segments_of_size_loop_idx < total_segments/size; 
            segments_of_size_loop_idx ++, past_seg_num += size){

        init_recv_seg(rank, size, count, past_seg_num, segment_size, total_segments,
                     false, &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
        update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
        // To get all finished results among process
        for(int step = 0; step < size - 1; step++){
            // Prepare message
            MPI_Request send_req, recv_req;
            // Non-blocking Send and Receive
            MPI_Isend(&recvbuf[start_idx], current_seg_size, MPI_FLOAT,
                    send_to, send_seg_idx, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(recv_seg, receive_seg_size, MPI_FLOAT,
                            recv_from, receive_seg_idx, MPI_COMM_WORLD, &recv_req);
            // Save the receive segment information
            int this_receive_seg_start_idx = receive_seg_start_idx;
            int this_receive_seg_size = receive_seg_size;
            // Overlap computation and communication
            update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
            // Process the data received
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            for (int i = 0; i < this_receive_seg_size; i++) {
                recvbuf[this_receive_seg_start_idx + i] += recv_seg[i];
            }
        }        

        // Exchange the finished segment, similar to broadcasting, 
        // achiving better performance than using MPI_Bcast.
        for (int step = 0; step < size - 1; step++) {
            // Prepare message
            MPI_Request send_req, recv_req;
            // Non-blocking Send and Receive
            MPI_Isend(&recvbuf[start_idx], current_seg_size, MPI_FLOAT,
                    send_to, send_seg_idx, MPI_COMM_WORLD, &send_req);
            MPI_Irecv(recv_seg, receive_seg_size, MPI_FLOAT,
                            recv_from, receive_seg_idx, MPI_COMM_WORLD, &recv_req);
            // Save the receive segment information
            int this_receive_seg_start_idx = receive_seg_start_idx;
            int this_receive_seg_size = receive_seg_size;
            // Overlap computation and communication
            update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
            // Process the data received
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
            for (int i = 0; i < this_receive_seg_size; i++) {
                recvbuf[this_receive_seg_start_idx + i] = recv_seg[i];
            }
        }
    }

    // The last iteration: size > number of segments need to be reduced.
    if(total_segments % size != 0) {
        init_recv_seg(rank, size, count, past_seg_num, segment_size, total_segments,
                     false ,&receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
        update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
        for(int step = 0; step < size - 1; step++){
            // Prepare message
            MPI_Request send_req, recv_req;

            // Non-blocking Send and Receive
            // Key point: Only send/receive the valid segments
            bool needSend = send_seg_idx < total_segments;
            bool needRecv = receive_seg_idx < total_segments;
            if (needSend) {
                MPI_Isend(&recvbuf[start_idx], current_seg_size, MPI_FLOAT,
                    send_to, send_seg_idx, MPI_COMM_WORLD, &send_req);
            }
            if (needRecv) {
                MPI_Irecv(recv_seg, receive_seg_size, MPI_FLOAT,
                            recv_from, receive_seg_idx, MPI_COMM_WORLD, &recv_req);
            }
            int this_receive_seg_start_idx = receive_seg_start_idx;
            int this_receive_seg_size = receive_seg_size;
            update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);

            if (needRecv) {
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
                // Process the data received
                for (int i = 0; i < this_receive_seg_size; i++) {
                    recvbuf[this_receive_seg_start_idx + i] += recv_seg[i];
                }
            }
        }

        // Exchange the finished segment using pipeline, similar to broadcasting
        for(int step = 0; step < size - 1; step++){
            // Prepare message
            MPI_Request send_req, recv_req;
            // Non-blocking Send and Receive
            bool needSend = send_seg_idx < total_segments;
            bool needRecv = receive_seg_idx < total_segments;
            if (needSend) {
                MPI_Isend(&recvbuf[start_idx], current_seg_size, MPI_FLOAT,
                    send_to, send_seg_idx, MPI_COMM_WORLD, &send_req);
                MPI_Wait(&send_req, MPI_STATUS_IGNORE);
            }
            if (needRecv) {
                MPI_Irecv(recv_seg, receive_seg_size, MPI_FLOAT,
                            recv_from, receive_seg_idx, MPI_COMM_WORLD, &recv_req);
            }
            int this_receive_seg_start_idx = receive_seg_start_idx;
            int this_receive_seg_size = receive_seg_size;
            // Computing
            update_send_recv_seg(size, count, past_seg_num, segment_size, total_segments,
                                &send_seg_idx, &start_idx, &current_seg_size, 
                                &receive_seg_idx, &receive_seg_start_idx, &receive_seg_size);
            // Process the data received
            if (needRecv) {
                MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
                
                for (int i = 0; i < this_receive_seg_size; i++) {
                    recvbuf[this_receive_seg_start_idx + i] = recv_seg[i];
                }
            }
        }
    }
    // Step 6: Cleanup
    free(send_seg);
    free(recv_seg);
}

int init_recv_seg(int rank, int size, int count, int past_seg_num, int segment_size, int total_segments,
                    bool isBcast, int *receive_seg_idx, int *receive_seg_start_idx, int *receive_seg_size) {
    int Bcast = isBcast? 1 : 0;
    *receive_seg_idx = (rank + size + Bcast) % size + past_seg_num;
    *receive_seg_start_idx = *receive_seg_idx * segment_size;
    *receive_seg_size = *receive_seg_idx >= total_segments - 1 ? count - *receive_seg_start_idx : segment_size;
}

int update_send_recv_seg(int size, int count, int past_seg_num, int segment_size, int total_segments,
                        int *send_seg_idx, int *start_idx, int *current_seg_size, 
                        int *receive_seg_idx, int *receive_seg_start_idx, int *receive_seg_size) {
    *send_seg_idx = *receive_seg_idx;
    *start_idx = *receive_seg_start_idx;
    *current_seg_size = *receive_seg_size;

    *receive_seg_idx -= 1;
    *receive_seg_start_idx -= segment_size;
    if (*receive_seg_idx < past_seg_num) {
        *receive_seg_idx += size;
        *receive_seg_start_idx = *receive_seg_idx * segment_size;
    }
    *receive_seg_size = *receive_seg_idx >= total_segments - 1? count - *receive_seg_start_idx : segment_size;
}