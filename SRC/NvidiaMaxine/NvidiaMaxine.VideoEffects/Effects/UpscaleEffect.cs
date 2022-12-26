// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-21-2022
//
// Last Modified By : Roman
// Last Modified On : 12-23-2022
// ***********************************************************************
// <copyright file="UpscaleEffect.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;

#if OPENCV
using OpenCvSharp;
#endif

using System;

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Upscale video effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />.
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    public class UpscaleEffect : BaseEffect
    {
        /// <summary>
        /// Gets or sets the new height. Width will be calculated automatically.
        /// </summary>
        /// <value>The new height.</value>
        public int NewHeight { get; set; }

        /// <summary>
        /// Gets or sets the strength.
        /// Strength 0 implies no enhancement, only upscaling.
        /// Strength 1 implies the maximum enhancement.
        /// </summary>
        /// <value>The strength.</value>
        public float Strength { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="UpscaleEffect" /> class.
        /// </summary>
        /// <param name="modelsDir">The models directory.</param>
        /// <param name="strength">The strength.</param>
        /// <param name="newHeight">The new height.</param>
        /// <param name="sourceImage">The source image.</param>
#if OPENCV
        public UpscaleEffect(string modelsDir, Mat sourceImage, float strength = 0.4f, int newHeight = 1080)
            : base(NvVFXFilterSelectors.NVVFX_FX_SR_UPSCALE, modelsDir, sourceImage)
#else
        public UpscaleEffect(string modelsDir, VideoFrame sourceImage, float strength = 0.4f, int newHeight = 1080) 
            : base(NvVFXFilterSelectors.NVVFX_FX_SR_UPSCALE, modelsDir, sourceImage)
#endif
        {
            Strength = strength;
            NewHeight = newHeight;
        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetF32(_handle, NvVFXParameterSelectors.NVVFX_STRENGTH, Strength));
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

            int dstWidth = _srcImg.Width * NewHeight / _srcImg.Height;
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
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Width, (uint)_srcImg.Height, NvCVImagePixelFormat.NVCV_RGBA, NvCVImageComponentType.NVCV_U8, NvCVLayout.NVCV_INTERLEAVED, NvCVMemSpace.NVCV_GPU, 32));

            // dst GPU
            _dstGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_dstImg.Width, (uint)_dstImg.Height, NvCVImagePixelFormat.NVCV_RGBA, NvCVImageComponentType.NVCV_U8, NvCVLayout.NVCV_INTERLEAVED, NvCVMemSpace.NVCV_GPU, 32));

            // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_srcImg, ref _srcVFX);

            // _dstVFX is an alias for _dstImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);

            CheckResult(AllocTempBuffers());

            return vfxErr;
        }
    }
}
