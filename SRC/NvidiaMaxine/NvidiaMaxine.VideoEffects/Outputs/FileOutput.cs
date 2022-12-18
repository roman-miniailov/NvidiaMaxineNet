﻿using CUDA;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Outputs
{
    public class FileOutput : IDisposable
    {
        private VideoWriter _writer = new VideoWriter();

        private bool disposedValue;

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

        public void WriteFrame(Mat frame)
        {
            _writer.Write(frame);
        }

        public void Finish()
        {
            _writer.Release();
        }

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

        ~FileOutput()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
