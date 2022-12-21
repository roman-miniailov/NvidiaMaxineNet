using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class SuperResolutionEffect : BaseEffect
    {
        /// <summary>
        /// Gets or sets the mode.
        /// </summary>
        /// <value>The mode.</value>
        public SuperResolutionEffectMode Mode { get; set; } = SuperResolutionEffectMode.LQSource;

        /// <summary>
        /// New height. Width will be calculated automatically.
        /// </summary>
        /// <value>The new height.</value>
        public int NewHeight { get; set; } = 1080;

        /// <summary>
        /// Initializes a new instance of the <see cref="SuperResolutionEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
        public SuperResolutionEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_SUPER_RES, modelsDir, sourceImage)
        {

        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MODE, (uint)Mode));
        }

        protected override NvCVStatus AllocBuffers(int width, int height)
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
            int dstWidth = _srcImg.Width * NewHeight / _srcImg.Height;
            _dstImg.Create(NewHeight, dstWidth, _srcImg.Type());
            if (_dstImg.Data == IntPtr.Zero)
            {
                return NvCVStatus.NVCV_ERR_MEMORY;
            }

            // src GPU
            _srcGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Cols, (uint)_srcImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //dst GPU
            _dstGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_dstImg.Cols, (uint)_dstImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            CheckResult(CheckScaleIsotropy(_srcGpuBuf, _dstGpuBuf));

            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            CheckResult(AllocTempBuffers());

            return vfxErr;
        }

        private NvCVStatus CheckScaleIsotropy(NvCVImage src, NvCVImage dst)
        {
            if (src.Width * dst.Height != src.Height * dst.Width)
            {
                Debug.WriteLine($"{src.Width}x{src.Height} --> {dst.Width}x{dst.Height}: different scale for width and height is not supported");
                return NvCVStatus.NVCV_ERR_RESOLUTION;
            }
            return NvCVStatus.NVCV_SUCCESS;
        }
    }
}
