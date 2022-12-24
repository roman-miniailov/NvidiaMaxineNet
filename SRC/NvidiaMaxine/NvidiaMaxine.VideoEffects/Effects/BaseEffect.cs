using CUDA;
using NvidiaMaxine.VideoEffects.API;

#if OPENCV
using OpenCvSharp;
using static OpenCvSharp.FileStorage;
#endif

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class BaseEffect : IDisposable
    {
        private string _id;

        private string _modelsDir;

        protected IntPtr _handle;

        protected IntPtr _state;

        protected IntPtr _stream;

        private bool disposedValue;

#if OPENCV
        protected Mat _srcImg;

        protected Mat _dstImg;
#else
        protected VideoFrame _srcImg;

        protected VideoFrame _dstImg;
#endif

        protected NvCVImage _srcGpuBuf;

        protected NvCVImage _dstGpuBuf;

        protected NvCVImage _srcVFX;

        protected NvCVImage _dstVFX;

        protected bool _useState = false;

        /// <summary>
        /// We use the same temporary buffer for source and dst, since it auto-shapes as needed.
        /// </summary>
        private NvCVImage _tmpVFX;

#if OPENCV
        public BaseEffect(string id, string modelsDir, Mat sourceImage)
#else
        public BaseEffect(string id, string modelsDir, VideoFrame sourceImage)
#endif
        {
            _id = id;
            _modelsDir = modelsDir;
            _srcImg = sourceImage;

            CreateEffect();
        }

        protected void CheckResult(NvCVStatus err)
        {
            if (err != NvCVStatus.NVCV_SUCCESS)
            {
                throw new Exception($"NvCVStatus exception. {err}");
            }
        }

        private NvCVStatus CreateEffect()
        {
            NvCVStatus vfxErr;
            vfxErr = NvVFXAPI.NvVFX_CreateEffect(_id, out _handle);
            CheckResult(vfxErr);

            if (!string.IsNullOrEmpty(_modelsDir))
            {
                vfxErr = NvVFXAPI.NvVFX_SetString(_handle, NvVFXParameterSelectors.NVVFX_MODEL_DIRECTORY, _modelsDir);
                CheckResult(vfxErr);
            }

            return vfxErr;
        }

        private void DestroyEffect()
        {
            if (_handle != IntPtr.Zero)
            {
                NvVFXAPI.NvVFX_DestroyEffect(_handle);
                _handle = IntPtr.Zero;
            }
            
            if (_stream != IntPtr.Zero)
            {
                CUDA.Runtime.API.cudaStreamDestroy(_stream);
                _stream = IntPtr.Zero;
            }

            // release state memory
            if (_state != IntPtr.Zero)
            {
                CUDA.Runtime.API.cudaFree(_state);
                _state = IntPtr.Zero;
            }

            _srcVFX.Destroy();
            _dstVFX.Destroy();
            _tmpVFX.Destroy();

            _dstImg.Dispose();

            _srcGpuBuf.Destroy();
            _dstGpuBuf.Destroy();
        }

#if OPENCV
        protected static void NVWrapperForCVMat(Mat cvIm, ref NvCVImage nvcvIm)
        {
            NvCVImagePixelFormat[] nvFormat = new[] { NvCVImagePixelFormat.NVCV_FORMAT_UNKNOWN, NvCVImagePixelFormat.NVCV_Y, NvCVImagePixelFormat.NVCV_YA, NvCVImagePixelFormat.NVCV_BGR, NvCVImagePixelFormat.NVCV_BGRA };
            NvCVImageComponentType[] nvType = new[] { NvCVImageComponentType.NVCV_U8, NvCVImageComponentType.NVCV_TYPE_UNKNOWN, NvCVImageComponentType.NVCV_U16, NvCVImageComponentType.NVCV_S16, NvCVImageComponentType.NVCV_S32,
              NvCVImageComponentType.NVCV_F32, NvCVImageComponentType.NVCV_F64, NvCVImageComponentType.NVCV_TYPE_UNKNOWN };

            nvcvIm.Pixels = cvIm.Data;
            nvcvIm.Width = (uint)cvIm.Width;
            nvcvIm.Height = (uint)cvIm.Height;
            nvcvIm.Pitch = (int)cvIm.Step(0);
            nvcvIm.PixelFormat = nvFormat[cvIm.Channels() <= 4 ? cvIm.Channels() : 0];
            nvcvIm.ComponentType = nvType[cvIm.Depth() & 7];
            nvcvIm.BufferBytes = 0;
            //nvcvIm.DeletePtr = null;
            //nvcvIm.DeleteProc = null;
            nvcvIm.PixelBytes = (byte)cvIm.Step(1);
            nvcvIm.ComponentBytes = (byte)cvIm.ElemSize1();
            nvcvIm.NumComponents = (byte)cvIm.Channels();
            nvcvIm.Planar = NvCVLayout.NVCV_CHUNKY;
            nvcvIm.GpuMem = NvCVMemSpace.NVCV_CPU;
            nvcvIm.Reserved1 = 0;
            nvcvIm.Reserved2 = 0;
        }
#else
        protected static void NVWrapperForCVMat(VideoFrame cvIm, ref NvCVImage nvcvIm)
        {
            nvcvIm.Pixels = cvIm.Data;
            nvcvIm.Width = (uint)cvIm.Width;
            nvcvIm.Height = (uint)cvIm.Height;
            nvcvIm.Pitch = (int)cvIm.Stride;
            nvcvIm.PixelFormat = cvIm.PixelFormat;
            nvcvIm.ComponentType = cvIm.ComponentType;
            nvcvIm.BufferBytes = 0;
            //nvcvIm.DeletePtr = null;
            //nvcvIm.DeleteProc = null;
            nvcvIm.PixelBytes = cvIm.PixelBytes;
            nvcvIm.ComponentBytes = cvIm.ComponentBytes;
            nvcvIm.NumComponents = cvIm.NumComponents;
            nvcvIm.Planar = NvCVLayout.NVCV_CHUNKY;
            nvcvIm.GpuMem = NvCVMemSpace.NVCV_CPU;
            nvcvIm.Reserved1 = 0;
            nvcvIm.Reserved2 = 0;
        }
#endif

        // Allocate one temp buffer to be used for input and output. Reshaping of the temp buffer in NvCVImage_Transfer() is done automatically,
        // and is very low overhead. We expect the destination to be largest, so we allocate that first to minimize reallocs probablistically.
        // Then we Realloc for the source to get the union of the two.
        // This could alternately be done at runtime by feeding in an empty temp NvCVImage, but there are advantages to allocating all memory at load time.
        protected virtual NvCVStatus AllocTempBuffers()
        {
            NvCVStatus vfxErr;
            vfxErr = NvCVImageAPI.NvCVImage_Alloc(ref _tmpVFX, _dstVFX.Width, _dstVFX.Height, _dstVFX.PixelFormat, _dstVFX.ComponentType, _dstVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);

            vfxErr = NvCVImageAPI.NvCVImage_Realloc(_tmpVFX, _srcVFX.Width, _srcVFX.Height, _srcVFX.PixelFormat, _srcVFX.ComponentType, _srcVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);

            return vfxErr;
        }

        protected virtual void ApplyEffect()
        {
        }

        public NvCVStatus Init(int width, int height)
        {
            _state = IntPtr.Zero;
            _stream = IntPtr.Zero;

            CheckResult(AllocBuffers(width, height));

            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE, ref _srcGpuBuf));
            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _dstGpuBuf));

            ApplyEffect();

            if (_useState)
            {
                uint stateSizeInBytes;
                CheckResult(NvVFXAPI.NvVFX_GetU32(_handle, NvVFXParameterSelectors.NVVFX_STATE_SIZE, out stateSizeInBytes));
                CUDA.Runtime.API.cudaMalloc(ref _state, stateSizeInBytes);
                CUDA.Runtime.API.cudaMemsetAsync(_state, 0, stateSizeInBytes, _stream); // <- stream BUG?

                IntPtr[] stateArray = new IntPtr[1];
                stateArray[0] = _state;
                IntPtr buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(typeof(IntPtr)) * stateArray.Length);
                Marshal.Copy(stateArray, 0, buffer, stateArray.Length);
                CheckResult(NvVFXAPI.NvVFX_SetObject(_handle, NvVFXParameterSelectors.NVVFX_STATE, buffer));
            }    
            else
            {
                //_stream = Marshal.AllocCoTaskMem(Marshal.SizeOf(typeof(IntPtr)));
                //Marshal.WriteIntPtr(_stream, IntPtr.Zero);

                //CheckResult(NvVFXAPI.NvVFX_CudaStreamCreate(out _stream));
                CheckResult(NvVFXAPI.NvVFX_SetCudaStream(_handle, NvVFXParameterSelectors.NVVFX_CUDA_STREAM, _stream));

                //var res = Marshal.ReadIntPtr(_stream);
            }

            CheckResult(NvVFXAPI.NvVFX_Load(_handle));

            return NvCVStatus.NVCV_SUCCESS;
        }

        protected virtual NvCVStatus AllocBuffers(int width, int height)
        {
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;

            if (_srcImg == null || _srcImg.Data == IntPtr.Zero)
            {
                // src CPU
#if OPENCV
                _srcImg = new Mat();
                _srcImg.Create(height, width, MatType.CV_8UC3);
#else
                _srcImg = new VideoFrame(width, height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);
#endif

                if (_srcImg.Data == IntPtr.Zero)
                {
                    return NvCVStatus.NVCV_ERR_MEMORY;
                }
            }
            else
            {
#if !OPENCV
                if (_srcImg.PixelFormat != NvCVImagePixelFormat.NVCV_BGR || _srcImg.ComponentType != NvCVImageComponentType.NVCV_U8)
                {
                    return NvCVStatus.NVCV_ERR_PARAMETER;
                }
#endif
            }

#if OPENCV
            _dstImg = new Mat();
            _dstImg.Create(_srcImg.Height, _srcImg.Width, _srcImg.Type());
#else
            _dstImg = new VideoFrame(_srcImg.Width, _srcImg.Height, _srcImg.PixelFormat, _srcImg.ComponentType);
#endif
            
            if (_dstImg.Data == IntPtr.Zero)
            {
                return NvCVStatus.NVCV_ERR_MEMORY;
            }

            // src GPU
            _srcGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Width, (uint)_srcImg.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //dst GPU
            _dstGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_dstImg.Width, (uint)_dstImg.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            CheckResult(AllocTempBuffers());

            return vfxErr;
        }

#if OPENCV
        public Mat Process()
#else
        public VideoFrame Process()
#endif
        {
            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcGpuBuf, 1.0f / 255.0f, _stream, _tmpVFX));
            CheckResult(NvVFXAPI.NvVFX_Run(_handle, 0));
            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstGpuBuf, _dstVFX, 255.0f, _stream, _tmpVFX));

            return _dstImg;
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                DestroyEffect();

                disposedValue = true;
            }
        }

        ~BaseEffect()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

    }
}
