using OpenCvSharp;

namespace NvidiaMaxine.VideoEffects
{
    public class VideoFrameEventArgs : EventArgs
    {
        public Mat Frame { get; set; }

        public VideoFrameEventArgs(Mat frame)
        {
            Frame = frame;
        }
    }
}
