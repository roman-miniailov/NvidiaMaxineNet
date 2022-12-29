// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-28-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-28-2022
// ***********************************************************************
// <copyright file="AudioCaptureSource.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NvidiaMaxine.AudioEffects.Sources
{
    /// <summary>
    /// Audio capture source.
    /// Implements the <see cref="IAudioSource" />
    /// Implements the <see cref="IDisposable" />.
    /// </summary>
    /// <seealso cref="IAudioSource" />
    /// <seealso cref="IDisposable" />
    public class AudioCaptureSource : IAudioSource, IDisposable
    {
        /// <summary>
        /// The source
        /// </summary>
        private WaveIn _source;

        /// <summary>
        /// The disposed value
        /// </summary>
        private bool disposedValue;

        /// <summary>
        /// Occurs when data available.
        /// </summary>
        public event EventHandler<float[]> DataAvailable;

        /// <summary>
        /// Occurs when complete.
        /// </summary>
        public event EventHandler<EventArgs> Complete;

        /// <summary>
        /// Enumerates this instance.
        /// </summary>
        /// <returns>IEnumerable&lt;System.String&gt;.</returns>
        public static IEnumerable<string> Enumerate()
        {
            int waveInDevices = WaveIn.DeviceCount;
            for (int waveInDevice = 0; waveInDevice < waveInDevices; waveInDevice++)
            {
                WaveInCapabilities deviceInfo = WaveIn.GetCapabilities(waveInDevice);
                yield return deviceInfo.ProductName;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="AudioCaptureSource"/> class.
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="bps">The BPS.</param>
        /// <param name="isFloat">if set to <c>true</c> [is float].</param>
        /// <exception cref="ArgumentException">Invalid device name</exception>
        public AudioCaptureSource(string name, int sampleRate, int channels, int bps, bool isFloat)
        {
            var sources = Enumerate();
            int id = sources.ToList().IndexOf(name);
            if (id < 0)
            {
                throw new ArgumentException("Invalid device name");
            }

            _source = new WaveIn();
            _source.DeviceNumber = id;

            if (isFloat)
            {
                _source.WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
            }
            else
            {
                _source.WaveFormat = new WaveFormat(sampleRate, bps, channels);
            }

            _source.DataAvailable += _source_DataAvailable;
            _source.RecordingStopped += _source_RecordingStopped;
        }

        /// <summary>
        /// Handles the RecordingStopped event of the _source control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="StoppedEventArgs"/> instance containing the event data.</param>
        private void _source_RecordingStopped(object sender, StoppedEventArgs e)
        {
            Complete?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Handles the DataAvailable event of the _source control.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="WaveInEventArgs"/> instance containing the event data.</param>
        private void _source_DataAvailable(object sender, WaveInEventArgs e)
        {
            float[] data = new float[e.BytesRecorded / 4];
            Buffer.BlockCopy(e.Buffer, 0, data, 0, e.BytesRecorded);
            DataAvailable?.Invoke(this, data);
        }

        /// <summary>
        /// Starts.
        /// </summary>
        /// <param name="size">The number of samples to read.</param>
        public void Start(int size)
        {
            _source.StartRecording();
        }

        /// <summary>
        /// Stops this instance.
        /// </summary>
        public void Stop()
        {
            _source.StopRecording();
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

                _source?.Dispose();
                _source = null;

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="AudioCaptureSource"/> class.
        /// </summary>
        ~AudioCaptureSource()
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
