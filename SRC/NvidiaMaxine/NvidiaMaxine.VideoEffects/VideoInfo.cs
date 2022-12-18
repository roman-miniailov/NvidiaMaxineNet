using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public class VideoInfo
    {
        public int Codec { get; set; }
        
        public int Width { get; set; }

        public int Height { get; set; }
        
        public double FrameRate { get; set; }    
        
        public long FrameCount { get; set; }

        public OpenCvSharp.Size Resolution { get { return new OpenCvSharp.Size(Width, Height); } }
    }
}
