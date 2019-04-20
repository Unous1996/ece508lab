#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 2048
#define BLOCK_SIZE_CIRCULAR 512
#define TILE_SIZE_CIRCULAR 2048

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}

/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

__device__ void merge_sequential_circular(float*A, int m, float*B, int n, float* C, int A_S_start, int B_S_start){
     int i = 0, j = 0, k = 0;
     while(i < m && j < n){
        int i_cir = (A_S_start + i) % TILE_SIZE_CIRCULAR;
        int j_cir = (B_S_start + j) % TILE_SIZE_CIRCULAR;
        if(A[i_cir] <= B[j_cir]){
            C[k++] = A[i_cir];
            i++;
        }
        else{
            C[k++] = B[j_cir];
            j++;
        }
     }

     if(i==m){
        for(; j < n; j++){
            int j_cir = (B_S_start + j) % TILE_SIZE_CIRCULAR;
            C[k++] = B[j_cir];
        }
     }
     else{
        for(; i < m; i++){
            int i_cir = (A_S_start + i) % TILE_SIZE_CIRCULAR;
            C[k++] = A[i_cir];
        }
     }
}


__device__ int co_rank(int k, float *A, int m, float *B, int n) {
    int i = k < m?k:m;
    int j = k - i;
    int i_low = 0>(k-n)?0:k-n;
    int j_low = 0>(k-m)?0:k-m;
    int delta;
    bool active=true;
    while(active){
        if (i > 0 && j < n && A[i-1] > B[j]){
            delta = ((i - i_low +1) >> 1);
            j_low = j;
            j += delta;
            i -= delta;
        }
        else if(j > 0 && i < m && B[j-1] >= A[i]){
            delta = ((j - j_low +1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else{
            active = false;
        }
    }
    return i;
}

__device__ int co_rank_circular(int k, float *A, int m, float *B, int n, int A_S_start, int B_S_start){
    int i = k < m?k:m;
    int j = k - i;
    int i_low = 0>(k-n)?0:k-n;
    int j_low = 0>(k-m)?0:k-m;
    int delta = 1;
    bool active=true;
    while(active){
        int i_cir = (A_S_start+i>=TILE_SIZE_CIRCULAR)?A_S_start+i-TILE_SIZE_CIRCULAR:A_S_start+i;
        int i_m_1_cir = (A_S_start+i-1>=TILE_SIZE_CIRCULAR)?A_S_start+i-TILE_SIZE_CIRCULAR-1:A_S_start+i-1;

        int j_cir = (B_S_start+j>=TILE_SIZE_CIRCULAR)?B_S_start+j-TILE_SIZE_CIRCULAR:B_S_start+j;
        int j_m_1_cir = (B_S_start+j-1>=TILE_SIZE_CIRCULAR)?B_S_start+j-TILE_SIZE_CIRCULAR-1:B_S_start+j-1;

        if(i > 0 && j < n && A[i_m_1_cir] > B[j_cir]){
            delta = ((i - i_low +1) >> 1);
            j_low = j;
            j += delta;
            i -= delta;
        } else if(j > 0 && i < m && B[j_m_1_cir] >= A[i_cir]) {
            delta = ((j - j_low +1) >> 1) ;
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int m=A_len, n=B_len;
    int k_curr = tid*ceil_div(m+n,blockDim.x*gridDim.x); 
    int k_next = min((tid+1) * ceil_div(m+n,blockDim.x*gridDim.x), m+n); 
    int i_curr = co_rank(k_curr, A, m, B, n); 
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr; int j_next = k_next - i_next;
    merge_sequential( &A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr] );
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    
    extern __shared__ float shareAB[];
    float* A_S = &shareAB[0]; 
    float* B_S = &shareAB[TILE_SIZE];  

    int m = A_len, n = B_len;  

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    int C_curr = bx * ceil_div(m+n,gridDim.x);
    int C_next = min((bx+1) * ceil_div(m+n, gridDim.x), m+n);

    int A_curr = co_rank(C_curr, A, m, B, n);
    int B_curr = C_curr - A_curr;
    int A_next = co_rank(C_next, A, m, B, n);
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int total_iteration = ceil_div(C_length, TILE_SIZE); 
    int C_completed = 0; 
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iteration) {
        for(int i=0; i<TILE_SIZE; i+=blockDim.x){
            if(i + tx < A_length - A_consumed) {
                A_S[i + tx] = A[A_curr + A_consumed + i + tx]; 
            }
        }

        for(int i=0; i<TILE_SIZE; i+=blockDim.x){
            if(i + tx < B_length - B_consumed) {
                B_S[i + tx] = B[B_curr + B_consumed + i + tx]; 
            }
        }

        __syncthreads();
        int c_curr = tx * (TILE_SIZE/blockDim.x); 
        int c_next = (tx+1) * (TILE_SIZE/blockDim.x);
    
        c_curr = c_curr<=(C_length-C_completed)?c_curr:C_length-C_completed;
        c_next = c_next<=(C_length-C_completed)?c_next:C_length-C_completed;

        int a_curr = co_rank(c_curr, A_S, min(TILE_SIZE, A_length-A_consumed), B_S, min(TILE_SIZE, B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(TILE_SIZE, A_length-A_consumed), B_S, min(TILE_SIZE, B_length-B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(&A_S[a_curr], a_next-a_curr, &B_S[b_curr], b_next-b_curr, &C[C_curr+C_completed+c_curr]);
        counter ++;
        C_completed += TILE_SIZE;
        A_consumed += co_rank(TILE_SIZE, A_S, TILE_SIZE, B_S, TILE_SIZE); 
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    
    extern __shared__ float shareAB[];
    float* A_S = &shareAB[0]; 
    float* B_S = &shareAB[TILE_SIZE_CIRCULAR];  

    int tx = threadIdx.x, bx = blockIdx.x;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = TILE_SIZE_CIRCULAR;
    int B_S_consumed = TILE_SIZE_CIRCULAR;

    int m = A_len, n = B_len;  

    int C_curr = bx * ceil_div(m+n,gridDim.x);
    int C_next = min((bx+1) * ceil_div(m+n, gridDim.x), m+n);

    int A_curr = co_rank(C_curr, A, m, B, n);
    int B_curr = C_curr - A_curr;
    int A_next = co_rank(C_next, A, m, B, n);
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;

    int total_iteration = ceil_div(C_length, TILE_SIZE_CIRCULAR); 

    int C_completed = 0; 
    int A_consumed = 0;
    int B_consumed = 0;

    while(counter < total_iteration) {
        for(int i = 0; i < A_S_consumed; i += blockDim.x){
            if(i + tx < A_length-A_consumed && i + tx < A_S_consumed){
                int load_index = A_S_start + (TILE_SIZE_CIRCULAR - A_S_consumed) + i + tx;
                load_index %= TILE_SIZE_CIRCULAR;
                A_S[load_index] = A[A_curr + A_consumed + i + tx]; 
            }
        }

        for(int i = 0; i < B_S_consumed; i += blockDim.x){
            if(i + tx < B_length-B_consumed && i + tx < B_S_consumed){
                int load_index = B_S_start + (TILE_SIZE_CIRCULAR - B_S_consumed) + i + tx;
                load_index %= TILE_SIZE_CIRCULAR;
                B_S[load_index] = B[B_curr + B_consumed + i + tx]; 
            }
        }

        __syncthreads();
        
        int c_curr = tx * (TILE_SIZE_CIRCULAR/blockDim.x); 
        int c_next = (tx+1) * (TILE_SIZE_CIRCULAR/blockDim.x);
    
        c_curr = c_curr<=(C_length-C_completed)?c_curr:C_length-C_completed;
        c_next = c_next<=(C_length-C_completed)?c_next:C_length-C_completed;

        int a_curr = co_rank_circular(c_curr, A_S, min(TILE_SIZE_CIRCULAR, A_length-A_consumed), B_S, min(TILE_SIZE_CIRCULAR, B_length-B_consumed),A_S_start,B_S_start);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next, A_S, min(TILE_SIZE_CIRCULAR, A_length-A_consumed), B_S, min(TILE_SIZE_CIRCULAR, B_length-B_consumed),A_S_start,B_S_start);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next-a_curr, B_S, b_next-b_curr, &C[C_curr+C_completed+c_curr], A_S_start+a_curr, B_S_start+b_curr);
        
        A_S_consumed = co_rank_circular(min(TILE_SIZE_CIRCULAR, C_length-C_completed), A_S, min(TILE_SIZE_CIRCULAR, A_length-A_consumed), B_S, min(TILE_SIZE_CIRCULAR, B_length-B_consumed), A_S_start, B_S_start);
        B_S_consumed = min(TILE_SIZE_CIRCULAR, C_length-C_completed) - A_S_consumed;
        
        A_consumed += A_S_consumed;
        C_completed += min(TILE_SIZE_CIRCULAR, C_length-C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start += A_S_consumed;
        if(A_S_start >= TILE_SIZE_CIRCULAR){
            A_S_start -= TILE_SIZE_CIRCULAR;
        }       

        B_S_start += B_S_consumed;
        if(B_S_start >= TILE_SIZE_CIRCULAR){
            B_S_start -= TILE_SIZE_CIRCULAR;
        }

        counter += 1;
        __syncthreads();
    }
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    size_t shemm_size;
    shemm_size = 2*TILE_SIZE*sizeof(float);
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE, shemm_size>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    size_t shemm_size;
    shemm_size = 2*TILE_SIZE_CIRCULAR*sizeof(float);
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE_CIRCULAR, shemm_size>>>(A, A_len, B, B_len, C);
}
