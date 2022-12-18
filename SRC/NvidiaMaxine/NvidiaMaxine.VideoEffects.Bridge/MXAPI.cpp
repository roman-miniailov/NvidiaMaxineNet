#include "MXAPI.h"

#include <cuda_runtime_api.h>

extern "C" __declspec(dllexport) int __stdcall mxtest(NvVFX_Handle eff, NvCVImage* srcGpuBuf, NvCVImage* dstGpuBuf, float strength, CUstream_st* stream, UINT32 * sizeInBytes)
{
    NvCV_Status status = NvCV_Status::NVCV_SUCCESS;
    void* state = nullptr;
    void* stateArray[1];
	
    status = NvVFX_SetImage(eff, NVVFX_INPUT_IMAGE, srcGpuBuf);
    status = NvVFX_SetImage(eff, NVVFX_OUTPUT_IMAGE, dstGpuBuf);

    status = NvVFX_SetF32(eff, NVVFX_STRENGTH, strength);

    unsigned int stateSizeInBytes;
	
    status = NvVFX_GetU32(eff, NVVFX_STATE_SIZE, &stateSizeInBytes);
    cudaMalloc(&state, stateSizeInBytes);
    cudaMemsetAsync(state, 0, stateSizeInBytes, stream);
    stateArray[0] = state;
    status = NvVFX_SetObject(eff, NVVFX_STATE, (void*)stateArray);

    status = NvVFX_Load(eff);

    *sizeInBytes = stateSizeInBytes;

    return (int)status;
}