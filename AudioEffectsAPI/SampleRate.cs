// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-29-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-29-2022
// ***********************************************************************
// <copyright file="SampleRate.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.AudioEffects
{
    /// <summary>
    /// Audio effect sample rate.
    /// </summary>
    public enum SampleRate
    {
        /// <summary>
        /// 8000 Hz.
        /// </summary>
        SR8000 = 8000,

        /// <summary>
        /// 16000 Hz.
        /// </summary>
        SR16000 = 16000,

        /// <summary>
        /// 48000 Hz.
        /// </summary>
        SR48000 = 48000,
    }
}
