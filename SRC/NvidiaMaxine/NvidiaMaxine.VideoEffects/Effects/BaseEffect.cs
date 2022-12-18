using CUDA;
using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static OpenCvSharp.FileStorage;

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class BaseEffect : IDisposable
    {
        private string _id;

        private string _modelsDir;

        protected IntPtr _handle;

        private IntPtr _state;

        private IntPtr _stream;

        private bool disposedValue;

        private Mat _srcImg;

        private Mat _dstImg;

        private NvCVImage _srcGpuBuf;

        private NvCVImage _dstGpuBuf;

        private NvCVImage _srcVFX;

        private NvCVImage _dstVFX;

        /// <summary>
        /// We use the same temporary buffer for source and dst, since it auto-shapes as needed.
        /// </summary>
        private NvCVImage _tmpVFX;

        public BaseEffect(string id, string modelsDir, Mat sourceImage)
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

        private void NVWrapperForCVMat(Mat cvIm, ref NvCVImage nvcvIm)
        {
            NvCVImagePixelFormat[] nvFormat = new[] { NvCVImagePixelFormat.NVCV_FORMAT_UNKNOWN, NvCVImagePixelFormat.NVCV_Y, NvCVImagePixelFormat.NVCV_YA, NvCVImagePixelFormat.NVCV_BGR, NvCVImagePixelFormat.NVCV_BGRA };
            NvCVImageComponentType[] nvType = new[] { NvCVImageComponentType.NVCV_U8, NvCVImageComponentType.NVCV_TYPE_UNKNOWN, NvCVImageComponentType.NVCV_U16, NvCVImageComponentType.NVCV_S16, NvCVImageComponentType.NVCV_S32,
              NvCVImageComponentType.NVCV_F32, NvCVImageComponentType.NVCV_F64, NvCVImageComponentType.NVCV_TYPE_UNKNOWN };

            nvcvIm.Pixels = cvIm.Data;
            nvcvIm.Width = (uint)cvIm.Cols;
            nvcvIm.Height = (uint)cvIm.Rows;
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

        // Allocate one temp buffer to be used for input and output. Reshaping of the temp buffer in NvCVImage_Transfer() is done automatically,
        // and is very low overhead. We expect the destination to be largest, so we allocate that first to minimize reallocs probablistically.
        // Then we Realloc for the source to get the union of the two.
        // This could alternately be done at runtime by feeding in an empty temp NvCVImage, but there are advantages to allocating all memory at load time.
        private NvCVStatus allocTempBuffers()
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

            CheckResult(allocBuffers(width, height));

            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE, ref _srcGpuBuf));
            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _dstGpuBuf));

            ApplyEffect();

            uint stateSizeInBytes;
            CheckResult(NvVFXAPI.NvVFX_GetU32(_handle, NvVFXParameterSelectors.NVVFX_STATE_SIZE, out stateSizeInBytes));
            CUDA.Runtime.API.cudaMalloc(ref _state, stateSizeInBytes);
            CUDA.Runtime.API.cudaMemsetAsync(_state, 0, stateSizeInBytes, _stream); // <- stream BUG?

            IntPtr[] stateArray = new IntPtr[1];
            stateArray[0] = _state;
            IntPtr buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(typeof(IntPtr)) * stateArray.Length);
            Marshal.Copy(stateArray, 0, buffer, stateArray.Length);
            CheckResult(NvVFXAPI.NvVFX_SetObject(_handle, NvVFXParameterSelectors.NVVFX_STATE, buffer));

            CheckResult(NvVFXAPI.NvVFX_Load(_handle));

            return NvCVStatus.NVCV_SUCCESS;
        }

        private NvCVStatus allocBuffers(int width, int height)
        {
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;

            if (_srcImg == null || _srcImg.Data == IntPtr.Zero)
            {
                // src CPU
                _srcImg = new Mat();
                _srcImg.Create(height, width, MatType.CV_8UC3);

                if (_srcImg.Data == IntPtr.Zero)
                {
                    return NvCVStatus.NVCV_ERR_MEMORY;
                }
            }

            _dstImg = new Mat();
            _dstImg.Create(_srcImg.Rows, _srcImg.Cols, _srcImg.Type());
            if (_dstImg.Data == IntPtr.Zero)
            {
                return NvCVStatus.NVCV_ERR_MEMORY;
            }

            // src GPU
            _srcGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Cols, (uint)_srcImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //dst GPU
            _dstGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_srcImg.Cols, (uint)_srcImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            CheckResult(allocTempBuffers());

            return vfxErr;
        }

        public Mat Process()
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
