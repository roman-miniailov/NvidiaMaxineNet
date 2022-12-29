// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-29-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-29-2022
// ***********************************************************************
// <copyright file="SampleRateHelper.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.AudioEffects
{
    /// <summary>
    /// Sample rate helper.
    /// </summary>
    public static class SampleRateHelper
    {
        /// <summary>
        /// Gets the size of the frame.
        /// </summary>
        /// <param name="sampleRate">The sample rate.</param>
        /// <returns>System.Int32.</returns>
        public static int GetFrameSize(this SampleRate sampleRate)
        {
            return sampleRate switch
            {
                SampleRate.SR8000 => 80,
                SampleRate.SR16000 => 160,
                SampleRate.SR48000 => 480,
                _ => 0
            };
        }
    }
}
