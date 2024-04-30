#include <iostream>
#include <vector>
#include <thread>
#include <nccl.h>
#include <cuda_runtime.h>

void ncclWorker(int rank, int numDevices, ncclUniqueId id) {

  ncclComm_t comm;
  cudaStream_t stream;
  const size_t dataSize = 1024;  // Example size
  float* buffer;

  // initialization
  cudaSetDevice(rank);
  ncclCommInitRank(&comm, numDevices, id, rank);
  cudaStreamCreate(&stream);
  cudaMalloc(&buffer, dataSize * sizeof(float));
  cudaMemset(buffer, 1, dataSize * sizeof(float));  

  // Setup CUDA Graph
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  ncclAllReduce(buffer, buffer, dataSize, ncclFloat, ncclSum, comm, stream);
  cudaStreamEndCapture(stream, &graph);
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
  cudaGraphDestroy(graph);

  // Luanch cuda_graph
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);

  // selectively update rank 0 cuda graph executor
  if (rank == 0) {
    cudaGraph_t newGraph;
    cudaGraphExecUpdateResultInfo updateResult;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    ncclAllReduce(buffer, buffer, dataSize, ncclFloat, ncclSum, comm, stream);
    cudaStreamEndCapture(stream, &newGraph);
    cudaGraphExecUpdate(instance, newGraph, &updateResult);

    if (updateResult.result == cudaGraphExecUpdateSuccess) {
      cudaStreamSynchronize(stream);
      std::cout << "graph update successfully" << std::endl;
    } else {
      std::cerr << "Graph update failed with result code: "
                << updateResult.result << std::endl;
    }
    cudaGraphDestroy(newGraph);
  }

  // Relaunch cuda graph executor
  cudaGraphLaunch(instance, stream);


  cudaGraphExecDestroy(instance);
  cudaFree(buffer);
  ncclCommDestroy(comm);
  cudaStreamDestroy(stream);

}

int main() {
  const int numDevices = 8;
  std::vector<std::thread> workers;

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  for (int i = 0; i < numDevices; ++i) {
    workers.push_back(std::thread(ncclWorker, i, numDevices, id));
  }

  for (auto& worker : workers) {
    worker.join();
  }
  return 0;
}
