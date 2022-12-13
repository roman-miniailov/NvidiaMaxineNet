using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{

    /// <summary>
    /// The format of pixels in an image.
    /// </summary>
    public enum NvCVImagePixelFormat
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
}
