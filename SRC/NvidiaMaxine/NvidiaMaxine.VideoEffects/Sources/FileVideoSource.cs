using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Sources
{
    public class FileVideoSource : IBaseSource, IDisposable
    {
        private VideoCapture _reader;

        private bool _stopFlag;
        
        private bool disposedValue;

        private bool _isRunning;

        private Mat _frame;

        public event EventHandler<VideoFrameEventArgs> FrameReady;

        public NvCVStatus Open(string filename)
        {
            _reader = new VideoCapture();
            _reader.Open(filename);
            if (!_reader.IsOpened())
            {
                throw new Exception("Failed to open video file: " + filename);
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

        public Mat GetBaseFrame()
        {
            return _frame;
        }
    
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
            });            
        }

        public void Stop()
        {
            _stopFlag = true;

            while (_isRunning)
            {
                Thread.Sleep(10);
            }
        }

        public void GetVideoInfo(out VideoInfo info)
        {
            Helpers.GetVideoInfo(_reader, false, out info);
        }

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

        ~FileVideoSource()
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
