// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-21-2022
//
// Last Modified By : Roman
// Last Modified On : 12-21-2022
// ***********************************************************************
// <copyright file="ArtifactReductionEffectMode.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Artifact reduction effect mode.
    /// </summary>
    public enum ArtifactReductionEffectMode : uint
    {
        /// <summary>
        /// Mode 0 removes lesser artifacts, preserves low gradient information better, and is suited for higher bitrate videos.
        /// </summary>
        HighBitrate,

        /// <summary>
        /// Mode 1 is better suited for lower bitrate videos.
        /// </summary>
        LowBitrate,
    }
}
