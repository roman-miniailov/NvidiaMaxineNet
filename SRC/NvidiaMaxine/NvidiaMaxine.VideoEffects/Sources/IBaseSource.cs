using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Sources
{
    public interface IBaseSource : IDisposable
    {
        public event EventHandler<VideoFrameEventArgs> FrameReady;

        public void Start();

        public void Stop();

        public void GetVideoInfo(out VideoInfo info);
    }
}
