// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-27-2022
// ***********************************************************************
// <copyright file="WAVFileReader.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using NAudio.Wave;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace NvidiaMaxine.AudioEffects.Sources
{
    /// <summary>
    /// WAV file reader.
    /// </summary>
    public class WAVFileSource : IAudioSource, IDisposable
    {
        /// <summary>
        /// The reader.
        /// </summary>
        private WaveFileReader _reader;

        /// <summary>
        /// The sample provider.
        /// </summary>
        private ISampleProvider _sampleProvider;

        /// <summary>
        /// The converter.
        /// </summary>
        private WaveFormatConversionStream _converter;

        /// <summary>
        /// The thread.
        /// </summary>
        private Thread _thread;

        /// <summary>
        /// The stop flag.
        /// </summary>
        private bool _stopFlag;

        /// <summary>
        /// The disposed value.
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
        /// Initializes a new instance of the <see cref="WAVFileSource"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="sampleRate">The sample rate.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="bps">The BPS.</param>
        /// <exception cref="FileNotFoundException">File not found</exception>
        public WAVFileSource(string filename, int sampleRate, int channels, int bps)
        {
            if (!File.Exists(filename))
            {
                throw new FileNotFoundException("File not found: " + filename);
            }

            _reader = new WaveFileReader(filename);

            bool convert = false;
            if (sampleRate > 0 && _reader.WaveFormat.SampleRate != sampleRate)
            {
                convert = true;
            }

            if (channels > 0 && _reader.WaveFormat.Channels != channels)
            {
                convert = true;
            }

            if (bps > 0 && _reader.WaveFormat.BitsPerSample != bps)
            {
                convert = true;
            }

            if (convert)
            {
                var waveFormat = new WaveFormat(sampleRate, bps, channels);

                _converter = new WaveFormatConversionStream(waveFormat, _reader);
                _sampleProvider = _converter.ToSampleProvider();
            }
            else
            {
                _sampleProvider = _reader.ToSampleProvider();
            }
        }

        /// <summary>
        /// Reads the entire wav file.
        /// </summary>
        /// <param name="align_samples">Align samples.</param>
        /// <returns>System.Single[].</returns>
        public float[] ReadWAVFile(int align_samples = -1)
        {
            long sampleCount = _converter.Length / (_converter.WaveFormat.BitsPerSample / 8);
            var buffer = new float[sampleCount];

            if (align_samples != -1 && align_samples > 0)
            {
                int num_frames = (int)(sampleCount / align_samples);
                if (sampleCount % align_samples > 0)
                {
                    num_frames++;
                }

                // allocate potentially a bigger sized buffer to align it to requested
                buffer = new float[num_frames * align_samples];
            }
            else
            {
                // data->resize(wave_file.GetNumSamples(), 0.f);
            }

            var read = _sampleProvider.Read(buffer, 0, buffer.Length);
            if (read == 0)
            {
                return new float[0];
            }

            //var read2 = _sampleProvider.Read(buffer, 0, buffer.Length);
            return buffer;
        }

        /// <summary>
        /// Reads.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="size">The size.</param>
        /// <returns>System.Int32.</returns>
        public int Read(float[] data, int offset, int size)
        {
            var read = _sampleProvider.Read(data, offset, size);
            return read;
        }

        /// <summary>
        /// Reads (async.).
        /// </summary>
        /// <param name="size">The size.</param>
        /// <returns>System.Int32.</returns>
        public Task<int> ReadAll(int size)
        {
            return Task.Run(() =>
            {
                var buffer = new float[size];
                var read = _sampleProvider.Read(buffer, 0, size);
                while (read > 0)
                {
                    DataAvailable?.Invoke(this, buffer);
                    read = _sampleProvider.Read(buffer, 0, size);
                }

                return read;
            });
        }

        /// <summary>
        /// Starts.
        /// </summary>
        /// <param name="size">The number of samples to read.</param>
        public void Start(int size)
        {
            _thread = new Thread(() =>
            {
                var buffer = new float[size];
                var read = _sampleProvider.Read(buffer, 0, size);
                while (read > 0 && !_stopFlag)
                {
                    DataAvailable?.Invoke(this, buffer);
                    read = _sampleProvider.Read(buffer, 0, size);
                }

                Complete?.Invoke(this, EventArgs.Empty);
            });

            _thread.Start();
        }

        /// <summary>
        /// Stops this instance.
        /// </summary>
        public void Stop()
        {
            _stopFlag = true;
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
                }

                Stop();

                _thread?.Join();
                _thread = null;

                _reader?.Dispose();
                _reader = null;

                _sampleProvider = null;

                _converter?.Dispose();
                _converter = null;

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="WAVFileSource"/> class.
        /// </summary>
        ~WAVFileSource()
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