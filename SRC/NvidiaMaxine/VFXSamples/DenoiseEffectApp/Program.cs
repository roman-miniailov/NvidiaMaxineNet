using OpenCvSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection.PortableExecutable;

namespace DenoiseEffectApp
{
    internal class Program
    {
        private static string DEFAULT_CODEC = "avc1";
        //#define DEFAULT_CODEC "H264"

        private static bool FLAG_debug = false;
        private static bool FLAG_verbose = false;
        private static bool FLAG_show = false;
        private static bool FLAG_progress = false;
        private static bool FLAG_webcam = false;
        private static float FLAG_strength = 0.0f;
        private static string FLAG_codec = DEFAULT_CODEC;
        private static string FLAG_camRes = "1280x720";
        private static string FLAG_inFile;
        private static string FLAG_outFile;
        private static string FLAG_outDir;
        private static string FLAG_modelDir;

        static void Main(string[] args)
        {


        }

        static void Usage()
        {
            Console.WriteLine("DenoiseEffectApp [args ...]\n" +
              "  where args is:\n" +
              "  --in_file=<path>           input file to be processed (can be an image but the best denoising performance is observed on videos)\n" +
              "  --webcam                   use a webcam as the input\n" +
              "  --out_file=<path>          output file to be written\n" +
              "  --show                     display the results in a window (for webcam, it is always true)\n" +
              "  --strength=<value>         strength of an effect [0-1]\n" +
              "  --model_dir=<path>         the path to the directory that contains the models\n" +
              "  --codec=<fourcc>           the fourcc code for the desired codec (default " + DEFAULT_CODEC + ")\n" +
              "  --progress                 show progress\n" +
              "  --verbose                  verbose output\n" +
              "  --debug                    print extra debugging information\n"
            );
        }

        static bool HasSuffix(string str, string suf)
        {
            return Path.GetExtension(str).ToLowerInvariant() == suf.ToLowerInvariant();
        }

        static bool HasOneOfTheseSuffixes(string str, string[] suffx)
        {
            foreach (var suf in suffx)
            {
                if (HasSuffix(str, suf))
                {
                    return true;
                }
            }

            return false;
        }

        static bool IsImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".bmp", ".jpg", ".jpeg", ".png" });
        }

        static bool IsLossyImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".jpg", ".jpeg" });
        }

        static string DurationString(double sc)
        {
            string buf;
            int hr, mn;
            hr = (int)(sc / 3600.0);
            sc -= hr * 3600.0;
            mn = (int)(sc / 60.0);
            sc -= mn * 60.0;
            buf = string.Format("%02d:%02d:%06.3f", hr, mn, sc);
            return buf;
        }

        static void GetVideoInfo(VideoCapture reader, string fileName, out VideoInfo info)
        {
            info.Codec = (int)reader.Get(VideoCaptureProperties.FourCC);
            info.Width = (int)reader.Get(VideoCaptureProperties.FrameWidth);
            info.Height = (int)reader.Get(VideoCaptureProperties.FrameHeight);
            info.FrameRate = (double)reader.Get(VideoCaptureProperties.Fps);
            info.FrameCount = (long)reader.Get(VideoCaptureProperties.FrameCount);

            if (FLAG_verbose)
                Console.WriteLine(
                  "       file \"%s\"\n" +
                  "      codec %.4s\n" +
                  "      width %4d\n" +
                  "     height %4d\n" +
                  " frame rate %.3f\n" +
                  "frame count %4lld\n" +
                  "   duration %s\n",
                  fileName, info.Codec, info.Width, info.Height, info.FrameRate, info.FrameCount,
                  DurationString(info.FrameCount / info.FrameRate)
                );
        }






    }
}