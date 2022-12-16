using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public static class Helpers
    {
        public static void GetVideoInfo(VideoCapture reader, string fileName, bool verbose, out VideoInfo info)
        {
            info.Codec = (int)reader.Get(VideoCaptureProperties.FourCC);
            info.Width = (int)reader.Get(VideoCaptureProperties.FrameWidth);
            info.Height = (int)reader.Get(VideoCaptureProperties.FrameHeight);
            info.FrameRate = (double)reader.Get(VideoCaptureProperties.Fps);
            info.FrameCount = (long)reader.Get(VideoCaptureProperties.FrameCount);

            if (verbose)
            {
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

        public static string DurationString(double sc)
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


        public static bool HasSuffix(string str, string suf)
        {
            return Path.GetExtension(str).ToLowerInvariant() == suf.ToLowerInvariant();
        }

        public static bool HasOneOfTheseSuffixes(string str, string[] suffx)
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

        public static bool IsImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".bmp", ".jpg", ".jpeg", ".png" });
        }

        public static bool IsLossyImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".jpg", ".jpeg" });
        }
    }
}
