using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Image
{
    /// <summary>
    /// Constants.
    /// </summary>
    public enum Constants
    {
        /// <summary>
        /// The Rec.601  YUV colorspace, typically used for SD.
        /// </summary>
        NVCV_601 = 0x00,

        /// <summary>
        ///  The Rec.709  YUV colorspace, typically used for HD.
        /// </summary>
        NVCV_709 = 0x01,

        /// <summary>
        ///  The Rec.2020 YUV colorspace.
        /// </summary>
        NVCV_2020 = 0x02,

        /// <summary>
        ///  The video range is [16, 235].
        /// </summary>
        NVCV_VIDEO_RANGE = 0x00,

        /// <summary>
        /// The video range is [ 0, 255].
        /// </summary>
        NVCV_FULL_RANGE = 0x04,

        /// <summary>
        /// The chroma is sampled at the same location as the luma samples horizontally.
        /// </summary>
        NVCV_CHROMA_COSITED = 0x00,

        /// <summary>
        /// The chroma is sampled between luma samples horizontally.
        /// </summary>
        NVCV_CHROMA_INTERSTITIAL = 0x08,

        /// <summary>
        /// The chroma is sampled at the same location as the luma samples horizontally and vertically.
        /// </summary>
        NVCV_CHROMA_TOPLEFT = 0x10,

        /// <summary>
        /// MPEG-2.
        /// </summary>
        NVCV_CHROMA_MPEG2 = NVCV_CHROMA_COSITED,

        /// <summary>
        /// MPEG-1.
        /// </summary>
        NVCV_CHROMA_MPEG1 = NVCV_CHROMA_INTERSTITIAL,

        /// <summary>
        /// JPEG.
        /// </summary>
        NVCV_CHROMA_JPEG = NVCV_CHROMA_INTERSTITIAL,

        /// <summary>
        /// H261.
        /// </summary>
        NVCV_CHROMA_H261 = NVCV_CHROMA_INTERSTITIAL,
    }
}
