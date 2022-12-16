using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenoiseEffectApp
{
    internal class Context
    {
        public static string DEFAULT_CODEC = "avc1";
        
        public bool Debug = false;
        public bool Verbose = false;
        public bool Show = false;
        public bool Progress = false;
        public bool Webcam = false;
        public float Strength = 0.0f;
        public string Codec = DEFAULT_CODEC;
        public string CamRes = "1280x720";
        public string InFile;
        public string OutFile;
        public string OutDir;
        public string ModelDir;
    }
}
