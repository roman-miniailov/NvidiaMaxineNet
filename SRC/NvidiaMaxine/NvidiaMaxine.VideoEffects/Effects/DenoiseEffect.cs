// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-19-2022
//
// Last Modified By : Roman
// Last Modified On : 12-24-2022
// ***********************************************************************
// <copyright file="DenoiseEffect.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;

#if OPENCV
using OpenCvSharp;
#endif

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Denoise effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />.
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />.
    public class DenoiseEffect : BaseEffect
    {
        /// <summary>
        /// Gets or sets the strength.
        /// The Strength of value 0 corresponds to a weak effect, which places a higher emphasis on texture preservation.
        /// The Strength of value 1 corresponds to a strong effect, which places a higher emphasis on noise removal.
        /// </summary>
        /// <value>The strength.</value>
        public float Strength { get; set; }

#if OPENCV
        /// <summary>
        /// Initializes a new instance of the <see cref="DenoiseEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models directory.</param>
        /// <param name="strength">The strength.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <exception cref="System.Exception">Denoise effects supports up to 1920x1080 resolution.</exception>
        public DenoiseEffect(string modelsDir, Mat sourceImage, float strength = 0.7f)
            : base(NvVFXFilterSelectors.NVVFX_FX_DENOISING, modelsDir, sourceImage)
#else
        /// <summary>
        /// Initializes a new instance of the <see cref="DenoiseEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <exception cref="System.Exception">Denoise effects supports up to 1920x1080 resolution.</exception>
        public DenoiseEffect(string modelsDir, VideoFrame sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_DENOISING, modelsDir, sourceImage)
#endif
        {
            if (sourceImage.Width > 1920 || sourceImage.Height > 1080)
            {
                throw new System.Exception("Denoise effects supports up to 1920x1080 resolution.");
            }

            Strength = strength;

            _useState = true;
        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetF32(_handle, NvVFXParameterSelectors.NVVFX_STRENGTH, Strength));
        }
    }
}
