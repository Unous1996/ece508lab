#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/*
void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                 unsigned int *nodeVisited, unsigned int *currLevelNodes,
                 unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                 unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }
}
*/

/******************************************************************************
 GPU kernels 
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes){
  
  unsigned int tx = threadIdx.x, bx = blockIdx.x; 
  unsigned int current_level_node_index = bx * blockDim.x + tx;
  if(current_level_node_index < *numCurrLevelNodes){
    unsigned int node = currLevelNodes[current_level_node_index];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      if (atomicAdd(&(nodeVisited[neighbor]),1) == 0) {
        nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor; 
      }
    }
  }
  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the global queue
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE
  __shared__ unsigned int nextLevelNodes_s[BQ_CAPACITY];    
  __shared__ unsigned int numNextLevelNodes_s, our_numNextLevelNodes;      

  if(threadIdx.x == 0){
    numNextLevelNodes_s = 0;    
  }

  __syncthreads();      
  const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;    
  if(tid < *numCurrLevelNodes) {         
    const unsigned int my_vertex = currLevelNodes[tid];         
    for(unsigned int i = nodePtrs[my_vertex]; i < nodePtrs[my_vertex + 1]; ++i){             
      const unsigned int was_visited = atomicExch(&(nodeVisited[nodeNeighbors[i]]), 1);             
      if(!was_visited){                               
        const unsigned int my_tail = atomicAdd(&numNextLevelNodes_s, 1);                 
        if(my_tail < BQ_CAPACITY){                     
          nextLevelNodes_s[my_tail] = nodeNeighbors[i];                 
        } 
        else{ // If full, add it to the global queue directly                     
          numNextLevelNodes_s = BQ_CAPACITY;                     
          const unsigned int my_global_tail = atomicAdd(numNextLevelNodes, 1);                     
          nextLevelNodes[my_global_tail] = nodeNeighbors[i];                 
        }               
      }         
    }     
  }
       
  __syncthreads();  
  if(threadIdx.x == 0) {         
    our_numNextLevelNodes = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);     
  }

  __syncthreads();      
  for(unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += blockDim.x) {         
    nextLevelNodes[our_numNextLevelNodes + i] = nextLevelNodes_s[i];     
  }
  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue

  // Store block queue in global queue
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses one queue per warp

  // Initialize shared memory queue

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the warp queue
  // If full, add it to the block queue
  // If full, add it to the global queue 

  // Calculate space for warp queue to go into block queue

  // Store warp queue in block queue
  // If full, add it to the global queue

  // Calculate space for block queue to go into global queue
  // Saturate block queue counter
  // Calculate space for global queue

  // Store block queue in global queue
}

/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
