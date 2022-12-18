#pragma once

#include "framework.h"

#include "nvVideoEffects.h"

extern "C" __declspec(dllexport) int __stdcall mxtest(NvVFX_Handle eff, NvCVImage* srcGpuBuf, NvCVImage* dstGpuBuf, float strength, CUstream_st* stream, UINT32 * sizeInBytes);

