using System;

namespace NvidiaMaxine.VideoEffects.Sources
{
    public interface IBaseSource : IDisposable
    {
        public event EventHandler<VideoFrameEventArgs> FrameReady;

        public event EventHandler<EventArgs> Complete;

        public void Start();

        public void Stop();

        public void GetVideoInfo(out VideoInfo info);

#if OPENCV
        public OpenCvSharp.Mat GetBaseFrame();
#else
        public VideoFrame GetBaseFrame();
#endif
    }
}
