// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-21-2022
//
// Last Modified By : Roman
// Last Modified On : 12-21-2022
// ***********************************************************************
// <copyright file="SuperResolutionEffectMode.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Super Resolution effect mode.
    /// </summary>
    public enum SuperResolutionEffectMode : uint
    {
        /// <summary>
        /// HQ source mode enhances less and removes more encoding artifacts and is suited for lower-quality videos.
        /// </summary>
        HQSource,

        /// <summary>
        /// LQ source mode enhances more and is suited for higher quality lossless videos.
        /// </summary>
        LQSource,
    }
}
