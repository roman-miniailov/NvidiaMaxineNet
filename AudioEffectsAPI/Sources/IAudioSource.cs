// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-28-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-28-2022
// ***********************************************************************
// <copyright file="IAudioSource.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;

namespace NvidiaMaxine.AudioEffects.Sources
{
    /// <summary>
    /// Interface IAudioSource.
    /// Extends the <see cref="IDisposable" />.
    /// </summary>
    /// <seealso cref="IDisposable" />
    public interface IAudioSource : IDisposable
    {
        /// <summary>
        /// Occurs when data available.
        /// </summary>
        public event EventHandler<float[]> DataAvailable;

        /// <summary>
        /// Occurs when complete.
        /// </summary>
        public event EventHandler<EventArgs> Complete;

        /// <summary>
        /// Starts.
        /// </summary>
        /// <param name="size">The number of samples to read.</param>
        public void Start(int size);

        /// <summary>
        /// Stops this instance.
        /// </summary>
        public void Stop();
    }
}
