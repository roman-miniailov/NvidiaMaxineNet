// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-19-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-24-2022
// ***********************************************************************
// <copyright file="CameraVideoSource.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************


#if OPENCV

using DirectShowLib;
using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Sources
{
    /// <summary>
    /// Camera video source.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Sources.IBaseSource" />
    /// Implements the <see cref="IDisposable" />
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Sources.IBaseSource" />
    /// <seealso cref="IDisposable" />
    public class CameraVideoSource : IBaseSource, IDisposable
    {
        private VideoCapture _reader;

        private bool _stopFlag;

        private bool disposedValue;

        private bool _isRunning;

        private Mat _frame;

        /// <summary>
        /// Occurs when new frame arrived.
        /// </summary>
        public event EventHandler<VideoFrameEventArgs> FrameReady;

        /// <summary>
        /// Occurs when playback completed.
        /// </summary>
        public event EventHandler<EventArgs> Complete;

        /// <summary>
        /// Opens the specified camera.
        /// </summary>
        /// <param name="cameraName">The camera name.</param>
        /// <returns>NvCVStatus.</returns>
        public NvCVStatus Open(string cameraName)
        {
            var devices = ListDevices();
            int id = devices.ToList().IndexOf(cameraName);
            if (id == -1)
            {
                return NvCVStatus.NVCV_ERR_GENERAL;
            }

            _reader = new VideoCapture();
            _reader.Open(id, VideoCaptureAPIs.DSHOW);
            if (!_reader.IsOpened())
            {
                throw new Exception($"Failed to open camera: {cameraName}");
            }

            _frame = new Mat();
            int width = (int)_reader.Get(VideoCaptureProperties.FrameWidth);
            int height = (int)_reader.Get(VideoCaptureProperties.FrameHeight);
            _frame.Create(height, width, MatType.CV_8UC3);

            if (_frame.Data == IntPtr.Zero)
            {
                return NvCVStatus.NVCV_ERR_MEMORY;
            }

            return NvCVStatus.NVCV_SUCCESS;
        }

        /// <summary>
        /// Gets the base frame.
        /// </summary>
        /// <returns>OpenCvSharp.Mat.</returns>
        public Mat GetBaseFrame()
        {
            return _frame;
        }

        /// <summary>
        /// Starts this instance.
        /// </summary>
        public void Start()
        {
            _stopFlag = false;
            _isRunning = false;

            Task.Run(() =>
            {
                _isRunning = true;

                while (!_stopFlag)
                {
                    _reader.Read(_frame);
                    if (_frame.Empty())
                    {
                        break;
                    }

                    FrameReady?.Invoke(this, new VideoFrameEventArgs(_frame));
                }

                _isRunning = false;

                Complete?.Invoke(this, new EventArgs());
            });
        }

        /// <summary>
        /// Stops this instance.
        /// </summary>
        public void Stop()
        {
            Task.Run(() =>
            {
                _stopFlag = true;

                while (_isRunning)
                {
                    Thread.Sleep(10);
                }
            });
        }

        /// <summary>
        /// Gets the video information.
        /// </summary>
        /// <param name="info">The information.</param>
        public void GetVideoInfo(out VideoInfo info)
        {
            Helpers.GetVideoInfo(_reader, false, out info);
        }

        /// <summary>
        /// Returns device list.
        /// </summary>
        /// <returns>List&lt;System.String&gt;.</returns>
        public static IEnumerable<string> ListDevices()
        {
            var devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice);
            foreach (var device in devices)
            {
                yield return device.Name;
            }
        }

        /// <summary>
        /// Disposes the specified disposing.
        /// </summary>
        /// <param name="disposing">The disposing.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                }

                _reader.Release();
                _reader.Dispose();

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes this instance.
        /// </summary>
        ~CameraVideoSource()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        /// <summary>
        /// Disposes this instance.
        /// </summary>
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

#endif