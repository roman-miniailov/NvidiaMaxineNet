// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-21-2022
//
// Last Modified By : Roman
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="ArtifactReductionEffect.cs" company="Roman Miniailov">
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
    /// Artifact Reduction effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />.
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    public class ArtifactReductionEffect : BaseEffect
    {
        /// <summary>
        /// Gets or sets the mode.
        /// </summary>
        /// <value>The mode.</value>
        public ArtifactReductionEffectMode Mode { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ArtifactReductionEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models directory.</param>
        /// <param name="mode">The mode.</param>
        /// <param name="sourceImage">The source image.</param>
#if OPENCV
        public ArtifactReductionEffect(string modelsDir, Mat sourceImage, ArtifactReductionEffectMode mode = ArtifactReductionEffectMode.LowBitrate)
            : base(NvVFXFilterSelectors.NVVFX_FX_ARTIFACT_REDUCTION, modelsDir, sourceImage)
#else
        public ArtifactReductionEffect(string modelsDir, VideoFrame sourceImage, ArtifactReductionEffectMode mode = ArtifactReductionEffectMode.LowBitrate) 
            : base(NvVFXFilterSelectors.NVVFX_FX_ARTIFACT_REDUCTION, modelsDir, sourceImage)
#endif
        {
            Mode = mode;
        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MODE, (uint)Mode));
        }
    }
}
