// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-24-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-24-2022
// ***********************************************************************
// <copyright file="AIGSEffectMode.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// AIGS effect mode.
    /// </summary>
    public enum AIGSEffectMode
    {
        /// <summary>
        /// The matte.
        /// </summary>
        Matte,

#if OPENCV
        /// <summary>
        /// The light.
        /// </summary>
        Light,
#endif

        /// <summary>
        /// The green.
        /// </summary>
        Green,

        /// <summary>
        /// The white.
        /// </summary>
        White,

        /// <summary>
        /// The none.
        /// </summary>
        None,

        /// <summary>
        /// The background.
        /// </summary>
        Background,

        /// <summary>
        /// The blur.
        /// </summary>
        Blur,
    }
}
