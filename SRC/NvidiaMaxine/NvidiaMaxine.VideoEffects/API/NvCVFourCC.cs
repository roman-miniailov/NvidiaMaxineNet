using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// FourCC/layout.
    /// </summary>
    public enum NvCVLayout : byte
    {
        /// <summary>
        /// All components of pixel(x,y) are adjacent (same as chunky) (default for non-YUV).
        /// </summary>
        NVCV_INTERLEAVED = 0,

        /// <summary>
        /// All components of pixel(x,y) are adjacent (same as interleaved).
        /// </summary>
        NVCV_CHUNKY = 0,

        /// <summary>
        /// The same component of all pixels are adjacent.
        /// </summary>
        NVCV_PLANAR = 1,

        /// <summary>
        /// [UYVY] Chunky 4:2:2 (default for 4:2:2).
        /// </summary>
        NVCV_UYVY = 2,

        /// <summary>
        /// [VYUY] Chunky 4:2:2.
        /// </summary>
        NVCV_VYUY = 4,

        /// <summary>
        /// [YUYV] Chunky 4:2:2.
        /// </summary>
        NVCV_YUYV = 6,

        /// <summary>
        /// [YVYU] Chunky 4:2:2.
        /// </summary>
        NVCV_YVYU = 8,

        /// <summary>
        /// [YUV] Chunky 4:4:4.
        /// </summary>
        NVCV_CYUV = 10,

        /// <summary>
        /// [YVU] Chunky 4:4:4.
        /// </summary>
        NVCV_CYVU = 12,

        /// <summary>
        /// [Y][U][V] Planar 4:2:2 or 4:2:0 or 4:4:4.
        /// </summary>
        NVCV_YUV = 3,

        /// <summary>
        /// [Y][V][U] Planar 4:2:2 or 4:2:0 or 4:4:4.
        /// </summary>
        NVCV_YVU = 5,

        /// <summary>
        /// [Y][UV] Semi-planar 4:2:2 or 4:2:0 (default for 4:2:0).
        /// </summary>
        NVCV_YCUV = 7,

        /// <summary>
        /// [Y][VU] Semi-planar 4:2:2 or 4:2:0.
        /// </summary>
        NVCV_YCVU = 9,

        /// <summary>
        /// [Y][U][V] Planar 4:2:0.
        /// </summary>
        NVCV_I420 = NVCV_YUV,

        /// <summary>
        /// [Y][U][V] Planar 4:2:0.
        /// </summary>
        NVCV_IYUV = NVCV_YUV,

        /// <summary>
        /// [Y][V][U] Planar 4:2:0.
        /// </summary>
        NVCV_YV12 = NVCV_YVU,

        /// <summary>
        /// [Y][UV] Semi-planar 4:2:0 (default for 4:2:0).
        /// </summary>
        NVCV_NV12 = NVCV_YCUV,

        /// <summary>
        /// [Y][VU] Semi-planar 4:2:0.
        /// </summary>
        NVCV_NV21 = NVCV_YCVU,

        /// <summary>
        /// [YUYV] Chunky 4:2:2.
        /// </summary>
        NVCV_YUY2 = NVCV_YUYV,

        /// <summary>
        /// [Y][U][V] Planar 4:4:4.
        /// </summary>
        NVCV_I444 = NVCV_YUV,

        /// <summary>
        /// [Y][U][V] Planar 4:4:4.
        /// </summary>
        NVCV_YM24 = NVCV_YUV,

        /// <summary>
        /// [Y][V][U] Planar 4:4:4.
        /// </summary>
        NVCV_YM42 = NVCV_YVU,

        /// <summary>
        /// [Y][UV] Semi-planar 4:4:4.
        /// </summary>
        NVCV_NV24 = NVCV_YCUV,

        /// <summary>
        /// [Y][VU] Semi-planar 4:4:4.
        /// </summary>
        NVCV_NV42 = NVCV_YCVU,
    }
}
