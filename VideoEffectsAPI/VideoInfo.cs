namespace NvidiaMaxine.VideoEffects
{
    public class VideoInfo
    {
        public int Codec { get; set; }
        
        public int Width { get; set; }

        public int Height { get; set; }
        
        public double FrameRate { get; set; }    
        
        public long FrameCount { get; set; }

#if OPENCV
        public OpenCvSharp.Size Resolution { get { return new OpenCvSharp.Size(Width, Height); } }
#else
        public System.Drawing.Size Resolution { get { return new System.Drawing.Size(Width, Height); } }
#endif
    }
}
