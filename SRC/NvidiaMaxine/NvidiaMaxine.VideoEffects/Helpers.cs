// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-16-2022
//
// Last Modified By : Roman
// Last Modified On : 12-22-2022
// ***********************************************************************
// <copyright file="Helpers.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

#if OPENCV
using OpenCvSharp;
#endif

using System;
using System.IO;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Class Helpers.
    /// </summary>
    public static class Helpers
    {
#if OPENCV
        /// <summary>
        /// Gets the video information.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="verbose">if set to <c>true</c> [verbose].</param>
        /// <param name="info">The information.</param>
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
#endif

        /// <summary>
        /// Convert duration to the string.
        /// </summary>
        /// <param name="sc">The sc.</param>
        /// <returns>System.String.</returns>
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


        /// <summary>
        /// Determines whether the specified string has suffix.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="suf">The suf.</param>
        /// <returns><c>true</c> if the specified string has suffix; otherwise, <c>false</c>.</returns>
        public static bool HasSuffix(string str, string suf)
        {
            return Path.GetExtension(str).ToLowerInvariant() == suf.ToLowerInvariant();
        }

        /// <summary>
        /// Determines whether has one of these suffixes.
        /// </summary>
        /// <param name="str">The string.</param>
        /// <param name="suffx">The suffx.</param>
        /// <returns><c>true</c> if the string has one of these suffixes; otherwise, <c>false</c>.</returns>
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

        /// <summary>
        /// Determines whether [is image file] [the specified string].
        /// </summary>
        /// <param name="str">The string.</param>
        /// <returns><c>true</c> if is image file; otherwise, <c>false</c>.</returns>
        public static bool IsImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".bmp", ".jpg", ".jpeg", ".png" });
        }

        /// <summary>
        /// Determines whether [is lossy image file] [the specified string].
        /// </summary>
        /// <param name="str">The string.</param>
        /// <returns><c>true</c> if is lossy image file; otherwise, <c>false</c>.</returns>
        public static bool IsLossyImageFile(string str)
        {
            return HasOneOfTheseSuffixes(str, new[] { ".jpg", ".jpeg" });
        }
    }
}
