#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  // INSERT KERNEL CODE HERE
  __shared__ float MdB[TILE_SZ_RATIO][TILE_SZ_B];
  
  int numOfPhases = ceil((n*1.0)/TILE_SZ_B);
  int numOfIterations = ceil((k*1.0)/TILE_SZ_RATIO);
  int bx = blockIdx.x, tx = threadIdx.x;
  int row = bx * blockDim.x + tx;

  for(int p = 0; p < numOfPhases; p++){
    float reg[TILE_SZ_RATIO];
    for(int i = 0; i < numOfIterations; i++){
      //Load B into the shared memory
      int shared_row_offset = tx / TILE_SZ_B;
      int shared_col_offset = tx % TILE_SZ_B;
      int start_row_B = i * TILE_SZ_RATIO;
      int start_col_B = p * TILE_SZ_B;

      
      if(start_row_B + shared_row_offset < k && start_col_B + shared_col_offset < n){
        MdB[shared_row_offset][shared_col_offset] = B(start_row_B + shared_row_offset,start_col_B + shared_col_offset);
      }
      else{
        MdB[shared_row_offset][shared_col_offset] = 0;
      }

      for(int j = 0; j < TILE_SZ_RATIO; j++){
        if(row < m && i * TILE_SZ_RATIO + j < k){
          reg[j] = A(row, i * TILE_SZ_RATIO + j);
        }
        else{
          reg[j] = 0;
        }
      }
      __syncthreads();

      
      for(int j = 0; j < TILE_SZ_RATIO; j++){
        for(int kol = 0; kol < TILE_SZ_B; kol++){
          if(row < m && start_col_B + kol < n){
            C(row,start_col_B + kol) += reg[j] * MdB[j][kol]; // kol is a integer between [0,TILE_SZ_B)
          }
        }
      }
      
      __syncthreads();
    }
  }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    dim3 dimBlock(TILE_SZ_A, 1, 1);
    //INSERT CODE HERE
    dim3 dimGrid(ceil((m*1.0)/TILE_SZ_A), 1, 1);
    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, A, B, C);
    //INSERT CODE HERE
}

