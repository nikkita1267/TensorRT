#include "siluPlugin.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void SiLUKernel(size_t input_size, const T* input, T* output) {
    size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= input_size) {
        return;
    }

    output[index] = input[index] / (1 + expf(-input[index]));
}

template <typename T>
pluginStatus_t inferenceSiLU(int size, const T* input, T* output, cudaStream_t stream) {
    const int blockSize = 512;
    const int gridSize = (size + blockSize - 1) / blockSize;
    SiLUKernel<T><<<gridSize, blockSize, 0, stream>>>(size, input, output);
    return STATUS_SUCCESS;
}

int SiLUPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    switch (mDataType)
    {
    case DataType::kFLOAT:
        return inferenceSiLU(batchSize * mBatchDim, (float*)inputs[0], (float*)outputs[0], stream);
    case DataType::kINT32:
        return inferenceSiLU(batchSize * mBatchDim, (int32_t*)inputs[0], (int32_t*)outputs[0], stream);
    case DataType::kINT8:
        return inferenceSiLU(batchSize * mBatchDim, (int8_t*)inputs[0], (int8_t*)outputs[0], stream);
    }
    return 1;
}
