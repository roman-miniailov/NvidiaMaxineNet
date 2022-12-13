using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects
{
    [StructLayout(LayoutKind.Sequential)]
    public struct NvCVImageS
    {
        /// <summary>
        /// The number of pixels horizontally in the image.
        /// </summary>
        public uint Width;

        /// <summary>
        /// The number of pixels  vertically  in the image.
        /// </summary>
        public uint Height;

        /// <summary>
        /// The byte stride between pixels vertically.
        /// </summary>
        public int Pitch;

        /// <summary>
        /// The format of the pixels in the image.
        /// </summary>
        public NvCVImagePixelFormat PixelFormat;

        /// <summary>
        /// The data type used to represent each component of the image.
        /// </summary>
        public NvCVImageComponentType ComponentType;

        /// <summary>
        ///  The number of bytes in a chunky pixel.
        /// </summary>
        public byte PixelBytes;

        /// <summary>
        /// The number of bytes in each pixel component.
        /// </summary>
        public byte ComponentBytes;

        /// <summary>
        /// The number of components in each pixel.
        /// </summary>
        public byte NumComponents;

        /// <summary>
        /// NVCV_CHUNKY, NVCV_PLANAR, NVCV_UYVY, ....
        /// </summary>
        public byte Planar;

        /// <summary>
        /// NVCV_CPU, NVCV_CPU_PINNED, NVCV_CUDA, NVCV_GPU.
        /// </summary>
        public byte GpuMem;

        /// <summary>
        /// An OR of colorspace, range and chroma phase.
        /// </summary>
        public byte Colorspace;

        /// <summary>
        /// For structure padding and future expansion. Set to 0.
        /// </summary>
        public byte Reserved1;

        /// <summary>
        /// For structure padding and future expansion. Set to 0.
        /// </summary>
        public byte Reserved2;

        /// <summary>
        /// Pointer to pixel(0,0) in the image.
        /// </summary>
        public IntPtr Pixels;

        /// <summary>
        /// Buffer memory to be deleted (can be NULL).
        /// </summary>
        public IntPtr DeletePtr;

        /// <summary>
        ///  Delete procedure to call rather than free().
        /// </summary>
        public IntPtr DeleteProc; // (* deleteProc) (void* p);

        /// <summary>
        /// The maximum amount of memory available through pixels.
        /// </summary>
        public ulong BufferBytes;
    };
}