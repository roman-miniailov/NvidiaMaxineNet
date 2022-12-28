using NvidiaMaxine.VideoEffects.API;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenoiseEffectApp
{
    internal enum Err
    {
        errQuit = +1,                         // Application errors
        errFlag = +2,
        errRead = +3,
        errWrite = +4,
        errNone = NvCVStatus.NVCV_SUCCESS,               // Video Effects SDK errors
        errGeneral = NvCVStatus.NVCV_ERR_GENERAL,
        errUnimplemented = NvCVStatus.NVCV_ERR_UNIMPLEMENTED,
        errMemory = NvCVStatus.NVCV_ERR_MEMORY,
        errEffect = NvCVStatus.NVCV_ERR_EFFECT,
        errSelector = NvCVStatus.NVCV_ERR_SELECTOR,
        errBuffer = NvCVStatus.NVCV_ERR_BUFFER,
        errParameter = NvCVStatus.NVCV_ERR_PARAMETER,
        errMismatch = NvCVStatus.NVCV_ERR_MISMATCH,
        errPixelFormat = NvCVStatus.NVCV_ERR_PIXELFORMAT,
        errModel = NvCVStatus.NVCV_ERR_MODEL,
        errLibrary = NvCVStatus.NVCV_ERR_LIBRARY,
        errInitialization = NvCVStatus.NVCV_ERR_INITIALIZATION,
        errFileNotFound = NvCVStatus.NVCV_ERR_FILE,
        errFeatureNotFound = NvCVStatus.NVCV_ERR_FEATURENOTFOUND,
        errMissingInput = NvCVStatus.NVCV_ERR_MISSINGINPUT,
        errResolution = NvCVStatus.NVCV_ERR_RESOLUTION,
        errUnsupportedGPU = NvCVStatus.NVCV_ERR_UNSUPPORTEDGPU,
        errWrongGPU = NvCVStatus.NVCV_ERR_WRONGGPU,
        errUnsupportedDriver = NvCVStatus.NVCV_ERR_UNSUPPORTEDDRIVER,
        errCudaMemory = NvCVStatus.NVCV_ERR_CUDA_MEMORY,       // CUDA errors
        errCudaValue = NvCVStatus.NVCV_ERR_CUDA_VALUE,
        errCudaPitch = NvCVStatus.NVCV_ERR_CUDA_PITCH,
        errCudaInit = NvCVStatus.NVCV_ERR_CUDA_INIT,
        errCudaLaunch = NvCVStatus.NVCV_ERR_CUDA_LAUNCH,
        errCudaKernel = NvCVStatus.NVCV_ERR_CUDA_KERNEL,
        errCudaDriver = NvCVStatus.NVCV_ERR_CUDA_DRIVER,
        errCudaUnsupported = NvCVStatus.NVCV_ERR_CUDA_UNSUPPORTED,
        errCudaIllegalAddress = NvCVStatus.NVCV_ERR_CUDA_ILLEGAL_ADDRESS,
        errCuda = NvCVStatus.NVCV_ERR_CUDA,
    };
}
