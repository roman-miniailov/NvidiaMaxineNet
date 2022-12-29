// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-28-2022
// ***********************************************************************
// <copyright file="WAVFileWriter.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NAudio.Wave;
using System;

namespace NvidiaMaxine.AudioEffects.Outputs
{
    /// <summary>
    /// WAV file writer.
    /// Implements the <see cref="IDisposable" />.
    /// </summary>
    /// <seealso cref="IDisposable" />
    public class WAVFileWriter : IDisposable
    {
        /// <summary>
        /// The writer.
        /// </summary>
        private WaveFileWriter _writer;

        /// <summary>
        /// The disposed value.
        /// </summary>
        private bool disposedValue;

        /// <summary>
        /// Initializes a new instance of the <see cref="WAVFileWriter"/> class.
        /// </summary>
        /// <param name="wavFile">The wav file.</param>
        /// <param name="samplesPerSec">The samples per sec.</param>
        /// <param name="numChannels">The number channels.</param>
        /// <param name="bitsPerSample">The bits per sample.</param>
        /// <param name="isFloat">if set to <c>true</c> [is float].</param>
        public WAVFileWriter(string wavFile, int samplesPerSec, int numChannels, int bitsPerSample, bool isFloat)
        {
            if (isFloat)
            {
                _writer = new WaveFileWriter(wavFile, WaveFormat.CreateIeeeFloatWaveFormat(samplesPerSec, numChannels));
            }
            else
            {
                _writer = new WaveFileWriter(wavFile, new WaveFormat(samplesPerSec, bitsPerSample, numChannels));
            }
        }

        /// <summary>
        /// Writes the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        public void Write(float[] data)
        {
            _writer.WriteSamples(data, 0, data.Length);
        }

        /// <summary>
        /// Writes the specified data.
        /// </summary>
        /// <param name="data">The data.</param>
        public void Write(byte[] data)
        {
            _writer.Write(data, 0, data.Length);
        }

        /// <summary>
        /// Finishes this instance.
        /// </summary>
        public void Finish()
        {
            _writer?.Flush();
            _writer?.Dispose();
            _writer = null;
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                _writer?.Dispose();
                _writer = null;

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="WAVFileWriter"/> class.
        /// </summary>
        ~WAVFileWriter()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
