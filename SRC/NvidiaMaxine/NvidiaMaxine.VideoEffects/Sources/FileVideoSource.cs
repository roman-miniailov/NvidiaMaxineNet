using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
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

        public event EventHandler<VideoFrameEventArgs> FrameReady;

        public void Open(string filename)
        {
            _reader = new VideoCapture();
            _reader.Open(filename);
            if (!_reader.IsOpened())
            {
                throw new Exception("Failed to open video file: " + filename);
            }
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
                    Mat frame = new Mat();
                    _reader.Read(frame);
                    if (frame.Empty())
                    {
                        break;
                    }

                    FrameReady?.Invoke(this, new VideoFrameEventArgs(frame));

                    frame.Dispose();
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
