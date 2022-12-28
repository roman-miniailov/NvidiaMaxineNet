// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-19-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-22-2022
// ***********************************************************************
// <copyright file="NvCVImage.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// NvCVImage
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NvCVImage
    {
        /// <summary>
        /// The number of pixels horizontally in the image.
        /// </summary>
        public uint Width = 0;

        /// <summary>
        /// The number of pixels  vertically  in the image.
        /// </summary>
        public uint Height = 0;

        /// <summary>
        /// The byte stride between pixels vertically.
        /// </summary>
        public int Pitch = 0;

        /// <summary>
        /// The format of the pixels in the image.
        /// </summary>
        public NvCVImagePixelFormat PixelFormat = NvCVImagePixelFormat.NVCV_FORMAT_UNKNOWN;

        /// <summary>
        /// The data type used to represent each component of the image.
        /// </summary>
        public NvCVImageComponentType ComponentType = NvCVImageComponentType.NVCV_TYPE_UNKNOWN;

        /// <summary>
        /// The number of bytes in a chunky pixel.
        /// </summary>
        public byte PixelBytes = 0;

        /// <summary>
        /// The number of bytes in each pixel component.
        /// </summary>
        public byte ComponentBytes = 0;

        /// <summary>
        /// The number of components in each pixel.
        /// </summary>
        public byte NumComponents = 0;

        /// <summary>
        /// NVCV_CHUNKY, NVCV_PLANAR, NVCV_UYVY, ....
        /// </summary>
        public NvCVLayout Planar = NvCVLayout.NVCV_INTERLEAVED;

        /// <summary>
        /// NVCV_CPU, NVCV_CPU_PINNED, NVCV_CUDA, NVCV_GPU.
        /// </summary>
        public NvCVMemSpace GpuMem = 0;

        /// <summary>
        /// An OR of colorspace, range and chroma phase.
        /// </summary>
        public byte Colorspace = 0;

        /// <summary>
        /// For structure padding and future expansion. Set to 0.
        /// </summary>
        public byte Reserved1 = 0;

        /// <summary>
        /// For structure padding and future expansion. Set to 0.
        /// </summary>
        public byte Reserved2 = 0;

        /// <summary>
        /// Pointer to pixel(0,0) in the image.
        /// </summary>
        public IntPtr Pixels = IntPtr.Zero;

        /// <summary>
        /// Buffer memory to be deleted (can be NULL).
        /// </summary>
        public IntPtr DeletePtr = IntPtr.Zero;

        /// <summary>
        /// Delete procedure to call rather than free().
        /// </summary>
        public IntPtr DeleteProc = IntPtr.Zero; // (* deleteProc) (void* p);

        /// <summary>
        /// The maximum amount of memory available through pixels.
        /// </summary>
        public ulong BufferBytes = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVImage" /> struct.
        /// </summary>
        public NvCVImage()
        {
            //var sz = Marshal.SizeOf(typeof(NvCVImage));
            NvCVImageAPI.NvCVImage_Alloc(
                ref this,
                0,
                0,
                NvCVImagePixelFormat.NVCV_FORMAT_UNKNOWN,
                NvCVImageComponentType.NVCV_TYPE_UNKNOWN,
                0,
                0,
                0);

            //NvCVImageAPI.NvCVImage_Alloc(
            //    ref this,
            //    640,
            //    480,
            //    NvCVImagePixelFormat.NVCV_BGR,
            //    NvCVImageComponentType.NVCV_F32,
            //    NvCVLayout.NVCV_PLANAR,
            //    NvCVMemSpace.NVCV_CPU,
            //    0);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVImage" /> struct.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="format">The format.</param>
        /// <param name="type">The type.</param>
        /// <param name="layout">The layout.</param>
        /// <param name="memSpace">The memory space.</param>
        /// <param name="alignment">The alignment.</param>
        public NvCVImage(
            uint width,
            uint height,
            NvCVImagePixelFormat format,
            NvCVImageComponentType type,
            NvCVLayout layout,
            NvCVMemSpace memSpace,
            uint alignment)
        {
            NvCVImageAPI.NvCVImage_Alloc(ref this, width, height, format, type, layout, memSpace, alignment);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVImage" /> struct.
        /// </summary>
        /// <param name="fullImg">The full img.</param>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public NvCVImage(NvCVImage fullImg, int x, int y, uint width, uint height)
        {
            NvCVImageAPI.NvCVImage_InitView(this, fullImg, x, y, width, height);
        }

        /// <summary>
        /// Copy subimage.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="srcX">The source x.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="dstX">The destination x.</param>
        /// <param name="dstY">The destination y.</param>
        /// <param name="wd">The width.</param>
        /// <param name="ht">The height.</param>
        /// <param name="stream">The stream.</param>
        /// <returns>NvCVStatus.</returns>
        public NvCVStatus CopyFrom(
            NvCVImage src,
            int srcX,
            int srcY,
            int dstX,
            int dstY,
            int wd,
            int ht,
            IntPtr stream)
        {
            NvCVRect2i srcRect = new NvCVRect2i(srcX, srcY, wd, ht);
            NvCVPoint2i dstPt = new NvCVPoint2i(dstX, dstY);

            var tmp = new NvCVImage();
            var res = NvCVImageAPI.NvCVImage_TransferRect(src, srcRect, this, dstPt, 1.0f, stream, tmp);
            tmp.Destroy();

            return res;
        }

        /// <summary>
        /// Copy image.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="stream">The stream.</param>
        /// <returns>NvCVStatus.</returns>
        public NvCVStatus CopyFrom(NvCVImage src, IntPtr stream)
        {
            var tmp = new NvCVImage();
            var res = NvCVImageAPI.NvCVImage_Transfer(src, this, 1.0f, stream, tmp);
            tmp.Destroy();

            return res;
        }

        /// <summary>
        /// Destroys this instance.
        /// </summary>
        public void Destroy()
        {
            NvCVImageAPI.NvCVImage_Dealloc(ref this);
            //NvCVImageAPI.NvCVImage_Destroy(ref this);
        }
    };
}