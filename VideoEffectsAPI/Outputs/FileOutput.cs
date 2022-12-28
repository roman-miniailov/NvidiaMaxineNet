// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-19-2022
//
// Last Modified By : Roman
// Last Modified On : 12-22-2022
// ***********************************************************************
// <copyright file="FileOutput.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

#if OPENCV

using OpenCvSharp;
using OpenCvSharp.Internal;
using System;
using System.Diagnostics;

namespace NvidiaMaxine.VideoEffects.Outputs
{
    /// <summary>
    /// OpenCV File output.
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public class FileOutput : IDisposable
    {
        private VideoWriter _writer = new VideoWriter();

        private bool disposedValue;

        /// <summary>
        /// Initializes a new instance of the <see cref="FileOutput"/> class.
        /// </summary>
        public FileOutput()
        {
            NativeMethods.ErrorHandlerDefault = ErrorHandler;
        }

        /// <summary>
        /// Error handler.
        /// </summary>
        /// <param name="status">The status.</param>
        /// <param name="funcName">Name of the function.</param>
        /// <param name="errMsg">The error.</param>
        /// <param name="fileName">Name of the file.</param>
        /// <param name="line">The line.</param>
        /// <param name="userData">The user data.</param>
        /// <returns>int.</returns>
        private int ErrorHandler(ErrorCode status, string funcName, string errMsg, string fileName, int line, IntPtr userData)
        {
            Debug.WriteLine($"OpenCV Error: {errMsg} in {funcName} at {fileName}:{line}");
            return 0;
        }

        /// <summary>
        /// Initializes the specified filename.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="resolution">The resolution.</param>
        /// <param name="frameRate">The frame rate.</param>
        /// <returns>bool.</returns>
        public bool Init(string filename, Size resolution, double frameRate)
        {
            bool ok = _writer.Open(filename, VideoWriter.FourCC("avc1"), frameRate, resolution);
            if (!ok)
            {
                Debug.WriteLine($"Cannot open {filename} for video writing.");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Writes the frame.
        /// </summary>
        /// <param name="frame">The frame.</param>
        public void WriteFrame(Mat frame)
        {
            _writer.Write(frame);
        }

        /// <summary>
        /// Finishes this instance.
        /// </summary>
        public void Finish()
        {
            _writer.Release();
        }

        /// <summary>
        /// Disposes this instance.
        /// </summary>
        /// <param name="disposing">The disposing.</param>
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
        /// Finalizes this instance.
        /// </summary>
        ~FileOutput()
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