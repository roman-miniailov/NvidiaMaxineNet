// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-19-2022
//
// Last Modified By : Roman
// Last Modified On : 12-23-2022
// ***********************************************************************
// <copyright file="NvCVImagePixelFormat.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.VideoEffects.API
{

    /// <summary>
    /// The format of pixels in an image.
    /// </summary>
    public enum NvCVImagePixelFormat : int
    {
        /// <summary>
        /// Unknown pixel format.
        /// </summary>
        NVCV_FORMAT_UNKNOWN = 0,

        /// <summary>
        /// Luminance (gray).
        /// </summary>
        NVCV_Y = 1,

        /// <summary>
        /// Alpha (opacity).
        /// </summary>
        NVCV_A = 2,

        /// <summary>
        /// { Luminance, Alpha }.
        /// </summary>
        NVCV_YA = 3,

        /// <summary>
        /// { Red, Green, Blue }.
        /// </summary>
        NVCV_RGB = 4,

        /// <summary>
        /// { Blue, Green, Red }.
        /// </summary>
        NVCV_BGR = 5,

        /// <summary>
        /// RGBA.
        /// </summary>
        NVCV_RGBA = 6,

        /// <summary>
        /// BGRA.
        /// </summary>
        NVCV_BGRA = 7,
        //#if RTX_CAMERA_IMAGE
        //  NVCV_YUV420          = 8,    //!< Luminance and subsampled Chrominance { Y, Cb, Cr }
        //  NVCV_YUV422          = 9,    //!< Luminance and subsampled Chrominance { Y, Cb, Cr }
        //#else // !RTX_CAMERA_IMAGE

        /// <summary>
        /// ARGB.
        /// </summary>
        NVCV_ARGB = 8,

        /// <summary>
        /// ABGR.
        /// </summary>
        NVCV_ABGR = 9,

        /// <summary>
        /// YUV420.
        /// </summary>
        NVCV_YUV420 = 10,

        /// <summary>
        /// YUV422.
        /// </summary>
        NVCV_YUV422 = 11,

        /// <summary>
        /// YUV444.
        /// </summary>
        NVCV_YUV444 = 12,
    }

    /// <summary>
    /// Class NvCVImagePixelFormatExtensions.
    /// </summary>
    public static class NvCVImagePixelFormatExtensions
    {
        /// <summary>
        /// Gets the channels count.
        /// </summary>
        /// <param name="format">The format.</param>
        /// <returns>System.Int32.</returns>
        public static int GetChannelsCount(this NvCVImagePixelFormat format)
        {
            switch (format)
            {
                case NvCVImagePixelFormat.NVCV_Y:
                    return 1;
                case NvCVImagePixelFormat.NVCV_A:
                    return 1;
                case NvCVImagePixelFormat.NVCV_YA:
                    return 2;
                case NvCVImagePixelFormat.NVCV_RGB:
                    return 3;
                case NvCVImagePixelFormat.NVCV_BGR:
                    return 3;
                case NvCVImagePixelFormat.NVCV_RGBA:
                    return 4;
                case NvCVImagePixelFormat.NVCV_BGRA:
                    return 4;
                case NvCVImagePixelFormat.NVCV_ARGB:
                    return 4;
                case NvCVImagePixelFormat.NVCV_ABGR:
                    return 4;
                case NvCVImagePixelFormat.NVCV_YUV420:
                    return 3;
                case NvCVImagePixelFormat.NVCV_YUV422:
                    return 3;
                case NvCVImagePixelFormat.NVCV_YUV444:
                    return 3;
                default:
                    return 0;
            }
        }
    }
}
