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
        public static void GetVideoInfo(VideoCapture reader, bool verbose, out VideoInfo info)
        {
            info = new VideoInfo();
            info.Codec = (int)reader.Get(VideoCaptureProperties.FourCC);
            info.Width = (int)reader.Get(VideoCaptureProperties.FrameWidth);
            info.Height = (int)reader.Get(VideoCaptureProperties.FrameHeight);
            info.FrameRate = (double)reader.Get(VideoCaptureProperties.Fps);
            info.FrameCount = (long)reader.Get(VideoCaptureProperties.FrameCount);

            if (verbose)
            {
                var dur = DurationString(info.FrameCount / info.FrameRate);
                Console.WriteLine(
                  $"      codec {info.Codec}\n" +
                  $"      width {info.Width}\n" +
                  $"     height {info.Height}\n" +
                  $" frame rate {info.FrameRate:F3}\n" +
                  $"frame count {info.FrameCount}\n" +
                  $"   duration {dur}\n");
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
            buf = $"{hr:D2}:{mn:D2}:{sc:F3}";
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
