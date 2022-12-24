// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-21-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-22-2022
// ***********************************************************************
// <copyright file="SuperResolutionEffect.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;

#if OPENCV
using OpenCvSharp;
#endif

using System;
using System.Diagnostics;

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Super resolution effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
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
        /// Initializes a new instance of the <see cref="SuperResolutionEffect" /> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
#if OPENCV
        public SuperResolutionEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_SUPER_RES, modelsDir, sourceImage)
#else
        public SuperResolutionEffect(string modelsDir, VideoFrame sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_SUPER_RES, modelsDir, sourceImage)
#endif
        {
        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MODE, (uint)Mode));
        }

        /// <summary>
        /// Allocs the buffers.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>NvCVStatus.</returns>
        protected override NvCVStatus AllocBuffers(int width, int height)
        {
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;

            if (_srcImg == null || _srcImg.Data == IntPtr.Zero)
            {
                // src CPU
#if OPENCV
                _srcImg = new Mat();
                _srcImg.Create(height, width, MatType.CV_8UC3);
#else
                _srcImg = new VideoFrame(height, width, NvCVImagePixelFormat.NVCV_RGB, NvCVImageComponentType.NVCV_U8);
#endif

                if (_srcImg.Data == IntPtr.Zero)
                {
                    return NvCVStatus.NVCV_ERR_MEMORY;
                }
            }

            int dstWidth = (int)Math.Round((double)_srcImg.Width * (double)NewHeight / (double)_srcImg.Height);

#if OPENCV
            _dstImg = new Mat();            
            _dstImg.Create(NewHeight, dstWidth, _srcImg.Type());
#else
            _dstImg = new VideoFrame(dstWidth, NewHeight, _srcImg.PixelFormat, _srcImg.ComponentType);
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

            CheckResult(CheckScaleIsotropy(_srcGpuBuf, _dstGpuBuf));

            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            CheckResult(AllocTempBuffers());

            return vfxErr;
        }

        /// <summary>
        /// Checks the scale isotropy.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="dst">The DST.</param>
        /// <returns>NvCVStatus.</returns>
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
