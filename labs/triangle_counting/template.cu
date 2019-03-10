#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <inttypes.h>

#include "template.hu"

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  // Determine the source and destination node for the edge
  unsigned int tx = threadIdx.x, bx = blockIdx.x;
  unsigned int edgeidx = bx * blockDim.x + tx;
  uint64_t result = 0;
  
  if(edgeidx < numEdges){
    unsigned int startNode = edgeSrc[edgeidx];
    unsigned int endNode = edgeDst[edgeidx];
    unsigned int u_ptr = rowPtr[startNode], v_ptr = rowPtr[endNode]; 
    unsigned int u_end = rowPtr[startNode+1], v_end = rowPtr[endNode+1];
    uint32_t w1 = edgeDst[u_ptr], w2 = edgeDst[v_ptr]; 
    while(u_ptr < u_end && v_ptr < v_end){
      if(w1 < w2){
        w1 = edgeDst[++u_ptr];
      }
      else if(w1 > w2){
        w2 = edgeDst[++v_ptr];
      }
      else{
        w1 = edgeDst[++u_ptr];
        w2 = edgeDst[++v_ptr];
        result++;
      }
    }
    //printf("result = %d\n", result);
    triangleCounts[edgeidx] = result;
  }
  
  // Use the row pointer array to determine the start and end of the neighbor list in the column index array

  // Determine how many elements of those two arrays are common
}



uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.

    //@@ launch the linear search kernel here
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(view.num_rows()/512.0));
    pangolin::Vector<uint64_t>::Vector TC(view.nnz(),0);
    // dim3 dimGrid (ceil(number of non-zeros / dimBlock.x))
    /*
    num_rows() returns the number of rows
    nnz() returns the number of non-zeros
    row_ptr() returns a pointer to the CSR row pointer array (length = num_rows() + 1 or 0 if num_rows() == 0).
    col_ind() returns a pointer to the CSR column index array (length = nnz()).
    row_ind() returns a pointer to the CSR row index array (length = nnz()).
    */
    kernel_tc<<<dimGrid, dimBlock>>>(TC.data(),view.row_ind(),view.col_ind(),view.row_ptr(),view.nnz());
    cudaDeviceSynchronize();

    uint64_t total = 0;
    
    for(unsigned int i = 0; i < view.nnz(); i++){
      total += TC[i];
    }

    printf("%" PRId64 "\n", total);
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
    return total;

  } else if (2 == mode) {
    // OPTIONAL. See README for more details
    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

    return total;
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}
