﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DenoiseEffectApp
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct VideoInfo
    {
        public int Codec;
        public int Width;
        public int Height;
        public double FrameRate;
        public long FrameCount;
    }
}
