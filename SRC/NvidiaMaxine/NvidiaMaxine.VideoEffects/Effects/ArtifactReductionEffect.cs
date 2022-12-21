// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : roman
// Created          : 12-21-2022
//
// Last Modified By : roman
// Last Modified On : 12-21-2022
// ***********************************************************************
// <copyright file="ArtifactReductionEffect.cs" company="NvidiaMaxine.VideoEffects">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Artifact Reduction effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    public class ArtifactReductionEffect : BaseEffect
    {
        /// <summary>
        /// Gets or sets the mode.
        /// </summary>
        /// <value>The mode.</value>
        public ArtifactReductionEffectMode Mode { get; set; } = ArtifactReductionEffectMode.LowBitrate;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArtifactReductionEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
        public ArtifactReductionEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_ARTIFACT_REDUCTION, modelsDir, sourceImage)
        {

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
