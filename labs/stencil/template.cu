#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 32
__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

    #define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
    #define Anext(i, j, k) Anext[((k)*ny + (j))*nx + (i)]

    __shared__ int MdA[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;

    int i = bx * TILE_SIZE + tx;
    int j = by * TILE_SIZE + ty;
        
    int bottom, current, top;    

    if(i >= 0 && i < nx && j >= 0 && j < ny){
        bottom = A0(i, j, 0);
        current = A0(i, j, 1);
        top = A0(i, j, 2);
    }

    MdA[ty][tx] = current;
    __syncthreads();

     
    for(int k = 1; k < nz - 1; k++){

      if(i > 0 && i < nx && j > 0 && j < ny){
          int north, south, west, east;
          west = (tx > 0) ? MdA[ty][tx-1] : A0(i-1,j,k);
          east = (tx < TILE_SIZE - 1) ? MdA[ty][tx+1] : A0(i+1,j,k);
          north = (ty > 0) ? MdA[ty-1][tx] : A0(i,j-1,k);
          south = (ty < TILE_SIZE - 1) ? MdA[ty+1][tx] : A0(i,j+1,k);

          Anext(i,j,k) = bottom + top + west + east + north + south - 6 * current;  
          __syncthreads();
      }

      bottom = current;

      if(i >= 0 && i < nx && j >= 0 && j < ny){
          MdA[i][j] = top;
          current = top;
          top = A0(i,j,k+2);
      } 

      __syncthreads();
      
   }


    #undef A0
    #undef Anext
    // INSERT KERNEL CODE HERE  
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
  dim3 dimGrid(ceil((nx*1.0)/TILE_SIZE),ceil((ny*1.0)/TILE_SIZE),1);
  INFO("Begin to launch kernel");
  kernel<<<dimGrid,dimBlock>>>(A0, Anext, nx, ny, nz);
  INFO("Finished launching kernel");
}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  bool debug = false;
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," + 
                                                   std::to_string(ny) + "," + 
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  if(debug == false){
      generate_data(hostA0.data(), nx, ny, nz);
  }
  else{
      generate_uniform_data(hostA0.data(), nx, ny, nz);
  }
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}

TEST_CASE("Convlayer", "[convlayer]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
