using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public static class NvCVImageAPI
    {
        private const string NvCVImageLib = "NVCVImage.dll";

        /// <summary>
        /// Get an error string corresponding to the given status code.
        /// </summary>
        /// <param name="code">The code.</param>
        /// <returns>System.String.</returns>
        [return: MarshalAs(UnmanagedType.LPStr)]
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern string NvCV_GetErrorStringFromCode(NvCVStatus code);

        /// <summary>
        /// Initialize an image. 
        /// </summary>
        /// <param name="im">The image.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="pitch">The byte stride between pixels vertically.</param>
        /// <param name="pixels">The pointer to the pixel buffer.</param>
        /// <param name="format">The format of the pixels.</param>
        /// <param name="type">The type of the components of the pixels.</param>
        /// <param name="layout">One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.</param>
        /// <param name="memSpace">Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if successful. NVCV_ERR_PIXELFORMAT if the pixel format is not yet accommodated.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Init(
            NvCVImage im, 
            uint width, 
            uint height, 
            int pitch, 
            IntPtr pixels,
            NvCVImagePixelFormat format, 
            NvCVImageComponentType type, 
            NvCVLayout layout, 
            NvCVMemSpace memSpace);

        /// <summary>
        /// Initialize a view into a subset of an existing image. No memory is allocated -- the fullImg buffer is used.
        /// </summary>
        /// <param name="subImg">The sub-image view into the existing full image.</param>
        /// <param name="fullImg">The existing full image.</param>
        /// <param name="x">The left edge of the sub-image, as coordinate of the full image.</param>
        /// <param name="y">The top edge of the sub-image, as coordinate of the full image.</param>
        /// <param name="width">The desired width of the subImage, in pixels.</param>
        /// <param name="height">The desired height of the subImage, in pixels.</param>
        /// <remarks>
        /// BUG! This does not work in general for planar or semi-planar formats, neither RGB nor YUV.
        /// However, it does work for all formats with the full image, to make a shallow copy, e.g.
        /// NvCVImage_InitView(&subImg, &fullImg, 0, 0, fullImage.width, fullImage.height).
        /// Cropping a planar or semi-planar image can be accomplished with NvCVImage_TransferRect().
        /// This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvCVImage_InitView(
            NvCVImage subImg, 
            NvCVImage fullImg, 
            int x, 
            int y,
            uint width, 
            uint height);

        /// <summary>
        /// Allocate memory for, and initialize an image. This assumes that the image data structure has nothing meaningful in it.
        /// </summary>
        /// <param name="im">The image to initialize.</param>
        /// <param name="width">The desired width of the image, in pixels.</param>
        /// <param name="height">The desired height of the image, in pixels.</param>
        /// <param name="format">The format of the pixels.</param>
        /// <param name="type">The type of the components of the pixels.</param>
        /// <param name="layout">One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.</param>
        /// <param name="memSpace">Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }.</param>
        /// <param name="alignment">
        /// The row byte alignment. Choose 0 or a power of 2. 
        /// 1: yields no gap whatsoever between scanlines;
        /// 0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
        /// Other common values are 16 or 32 for cache line size.
        /// </param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS - if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT - if the pixel format is not accommodated.
        /// NVCV_ERR_MEMORY - if there is not enough memory to allocate the buffer. 
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Alloc(
            ref NvCVImage im, 
            uint width, 
            uint height, 
            NvCVImagePixelFormat format,
            NvCVImageComponentType type, 
            NvCVLayout layout, 
            NvCVMemSpace memSpace, 
            uint alignment);

        /// <summary>
        /// Reallocate memory for, and initialize an image. This assumes that the image is valid. 
        /// It will check bufferBytes to see if enough memory is already available, and will reshape rather than realloc if true.
        /// Otherwise, it will free the previous buffer and reallocate a new one.
        /// </summary>
        /// <param name="im">The image to initialize.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="format">The format of the pixels.</param>
        /// <param name="type">The type of the components of the pixels.</param>
        /// <param name="layout">One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.</param>
        /// <param name="memSpace">Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }.</param>
        /// <param name="alignment">
        /// The row byte alignment. Choose 0 or a power of 2.
        /// 1: yields no gap whatsoever between scanlines;
        /// 0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
        /// Other common values are 16 or 32 for cache line size.
        /// </param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS - if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT - if the pixel format is not accommodated.
        /// NVCV_ERR_MEMORY - if there is not enough memory to allocate the buffer.
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Realloc(
            NvCVImage im, 
            uint width, 
            uint height, 
            NvCVImagePixelFormat format,
            NvCVImageComponentType type, 
            NvCVLayout layout,
            NvCVMemSpace memSpace,
            uint alignment);

        /// <summary>
        /// Deallocate the image buffer from the image. The image is not deallocated.
        /// </summary>
        /// <param name="im">The image whose buffer is to be deallocated..</param>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvCVImage_Dealloc(NvCVImage im);

        /// <summary>
        /// Deallocate the image buffer from the image asynchronously on the specified stream. The image is not deallocated.
        /// </summary>
        /// <param name="im">The image whose buffer is to be deallocated.</param>
        /// <param name="stream">The CUDA stream on which the image buffer is to be deallocated.</param>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvCVImage_DeallocAsync(NvCVImage im, IntPtr stream);

        /// <summary>
        /// Allocate a new image, with storage (C-style constructor).
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="format">The format of the pixels.</param>
        /// <param name="type">The type of the components of the pixels.</param>
        /// <param name="layout">One of { NVCV_CHUNKY, NVCV_PLANAR } or one of the YUV layouts.</param>
        /// <param name="memSpace">Location of the buffer: one of { NVCV_CPU, NVCV_CPU_PINNED, NVCV_GPU, NVCV_CUDA }.</param>
        /// <param name="alignment">
        /// The row byte alignment. Choose 0 or a power of 2.
        /// 1: yields no gap whatsoever between scanlines;
        /// 0: default alignment: 4 on CPU, and cudaMallocPitch's choice on GPU.
        /// Other common values are 16 or 32 for cache line size.
        /// </param>
        /// <param name="image">Will be a pointer to the new image if successful; otherwise NULL.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS - if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT - if the pixel format is not accommodated.
        /// NVCV_ERR_MEMORY - if there is not enough memory to allocate the buffer. 
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Create(
            uint width, 
            uint height,
            NvCVImagePixelFormat format,
            NvCVImageComponentType type, 
            NvCVLayout layout, 
            NvCVMemSpace memSpace, 
            uint alignment,
            out NvCVImage image);

        /// <summary>
        /// Deallocate the image allocated with NvCVImage_Create() (C-style destructor).
        /// </summary>
        /// <param name="im">The image.</param>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvCVImage_Destroy(NvCVImage im);

        /// <summary>
        /// Get offsets for the components of a pixel format. These are not byte offsets, but component offsets.
        /// </summary>
        /// <param name="format">The pixel format to be interrogated.</param>
        /// <param name="rOff">The place to store the offset for the red channel (can be NULL).</param>
        /// <param name="gOff">The place to store the offset for the green channel (can be NULL).</param>
        /// <param name="bOff">The place to store the offset for the blue channel (can be NULL).</param>
        /// <param name="aOff">The place to store the offset for the alpha channel (can be NULL).</param>
        /// <param name="yOff">The place to store the offset for the luminance channel (can be NULL).</param>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvCVImageComponentOffsets(
            NvCVImagePixelFormat format, 
            out int rOff,
            out int gOff, 
            out int bOff,
            out int aOff,
            out int yOff);
        
        /// <summary>
        /// Transfer one image to another, with a limited set of conversions.
        /// </summary>
        /// <param name="src">The source image.</param>
        /// <param name="dst">The  destination image.</param>
        /// <param name="scale">
        /// The scale factor that can be applied when one (but not both) of the images
        /// is based on floating-point components; this parameter is ignored when all image components
        /// are represented with integer data types, or all image components are represented with
        /// floating-point data types.
        /// </param>
        /// <param name="stream">The stream on which to perform the copy. This is ignored if both images reside on the CPU.</param>
        /// <param name="tmp">
        /// The temporary buffer that is sometimes needed when transferring images between the CPU and GPU in either direction 
        /// (can be empty or NULL). It has the same characteristics as the CPU image, but it resides on the GPU.
        /// </param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS - if successful.
        /// NVCV_ERR_PIXELFORMAT - if one of the pixel formats is not accommodated.
        /// NVCV_ERR_CUDA - if a CUDA error has occurred.
        /// NVCV_ERR_GENERAL - if an otherwise unspecified error has occurred.
        /// </returns>
        /// <remarks>
        /// If any of the images resides on the GPU, it may run asynchronously,
        /// so cudaStreamSynchronize() should be called if it is necessary to run synchronously.
        /// The following table indicates (with X) the currently-implemented conversions:
        ///    +-------------------+-------------+-------------+-------------+-------------+
        ///    |                   |  u8 --> u8  |  u8 --> f32 | f32 --> u8  | f32 --> f32 |
        ///    +-------------------+-------------+-------------+-------------+-------------+
        ///    | Y      --> Y      |      X      |             |      X      |      X      |
        ///    | Y      --> A      |      X      |             |      X      |      X      |
        ///    | Y      --> RGB    |      X      |      X      |      X      |      X      |
        ///    | Y      --> RGBA   |      X      |      X      |      X      |      X      |
        ///    | A      --> Y      |      X      |             |      X      |      X      |
        ///    | A      --> A      |      X      |             |      X      |      X      |
        ///    | A      --> RGB    |      X      |      X      |      X      |      X      |
        ///    | A      --> RGBA   |      X      |             |             |             |
        ///    | RGB    --> Y      |      X      |      X      |             |             |
        ///    | RGB    --> A      |      X      |      X      |             |             |
        ///    | RGB    --> RGB    |      X      |      X      |      X      |      X      |
        ///    | RGB    --> RGBA   |      X      |      X      |      X      |      X      |
        ///    | RGBA   --> Y      |      X      |      X      |             |             |
        ///    | RGBA   --> A      |      X      |             |             |             |
        ///    | RGBA   --> RGB    |      X      |      X      |      X      |      X      |
        ///    | RGBA   --> RGBA   |      X      |      X      |      X      |      X      |
        ///    | RGB    --> YUV420 |      X      |             |      X      |             |
        ///    | RGBA   --> YUV420 |      X      |             |      X      |             |
        ///    | RGB    --> YUV422 |      X      |             |      X      |             |
        ///    | RGBA   --> YUV422 |      X      |             |      X      |             |
        ///    | RGB    --> YUV444 |      X      |             |      X      |             |
        ///    | RGBA   --> YUV444 |      X      |             |      X      |             |
        ///    | YUV420 --> RGB    |      X      |      X      |             |             |
        ///    | YUV420 --> RGBA   |      X      |      X      |             |             |
        ///    | YUV422 --> RGB    |      X      |      X      |             |             |
        ///    | YUV422 --> RGBA   |      X      |      X      |             |             |
        ///    | YUV444 --> RGB    |      X      |      X      |             |             |
        ///    | YUV444 --> RGBA   |      X      |      X      |             |             |
        ///    +-------------------+-------------+-------------+-------------+-------------+
        /// where
        /// * Either source or destination can be CHUNKY or PLANAR.
        /// * Either source or destination can reside on the CPU or the GPU.
        /// * The RGB components are in any order (i.e. RGB or BGR; RGBA or BGRA).
        /// * For RGBA (or BGRA) destinations, most implementations do not change the alpha channel, so it is recommended to
        ///   set it at initialization time with [cuda]memset(im.pixels, -1, im.pitch * im.height) or
        ///   [cuda]memset(im.pixels, -1, im.pitch * im.height * im.numComponents) for chunky and planar images respectively.
        /// * YUV requires that the colorspace field be set manually prior to Transfer, e.g. typical for layout=NVCV_NV12:
        ///   image.colorspace = NVCV_709 | NVCV_VIDEO_RANGE | NVCV_CHROMA_INTSTITIAL;
        /// * There are also RGBf16-->RGBf32 and RGBf32-->RGBf16 transfers.
        /// * Additionally, when the src and dst formats are the same, all formats are accommodated on CPU and GPU,
        ///   and this can be used as a replacement for cudaMemcpy2DAsync() (which it utilizes). This is also true for YUV,
        ///   whose src and dst must share the same format, layout and colorspace.
        ///
        /// When there is some kind of conversion AND the src and dst reside on different processors (CPU, GPU),
        /// it is necessary to have a temporary GPU buffer, which is reshaped as needed to match the characteristics
        /// of the CPU image. The same temporary image can be used in subsequent calls to NvCVImage_Transfer(),
        /// regardless of the shape, format or component type, as it will grow as needed to accommodate
        /// the largest memory requirement. The recommended usage for most cases is to supply an empty image
        /// as the temporary; if it is not needed, no buffer is allocated. NULL can be supplied as the tmp
        /// image, in which case an ephemeral buffer is allocated if needed, with resultant
        /// performance degradation for image sequences.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Transfer(
             NvCVImage src, 
             NvCVImage dst, 
             float scale, 
             IntPtr stream, 
             NvCVImage tmp);

        /// <summary>
        /// Transfer a rectangular portion of an image. See NvCVImage_Transfer() for the pixel format combinations that are implemented.
        /// </summary>
        /// <param name="src">The source image.</param>
        /// <param name="srcRect">The subRect of the src to be transferred (NULL implies the whole image).</param>
        /// <param name="dst">The destination image.</param>
        /// <param name="dstPt">The location to which the srcRect is to be copied (NULL implies (0,0)).</param>
        /// <param name="scale">The scale factor applied to the magnitude during transfer, typically 1, 255 or 1/255.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="tmp">The staging image.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was completed successfully.
        /// </returns>
        /// <remarks>The actual transfer region may be smaller, because the rects are clipped against the images.</remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_TransferRect(
            NvCVImage src,
            NvCVRect2i srcRect,
            NvCVImage dst,
            NvCVPoint2i dstPt,
            float scale,
            IntPtr stream,
            NvCVImage tmp);

        /// <summary>
        /// Transfer from a YUV image. YUVu8 --> RGBu8 and YUVu8 --> RGBf32 are currently available.
        /// </summary>
        /// <param name="y">The pointer to pixel(0,0) of the luminance channel.</param>
        /// <param name="yPixBytes">The byte stride between y pixels horizontally.</param>
        /// <param name="yPitch">The byte stride between y pixels vertically.</param>
        /// <param name="u">The pointer to pixel(0,0) of the u (Cb) chrominance channel.</param>
        /// <param name="v">The pointer to pixel(0,0) of the v (Cr) chrominance channel.</param>
        /// <param name="uvPixBytes">The byte stride between u or v pixels horizontally.</param>
        /// <param name="uvPitch">The byte stride between u or v pixels vertically.</param>
        /// <param name="yuvFormat">The yuv format.</param>
        /// <param name="yuvType">Type of the yuv.</param>
        /// <param name="yuvColorSpace">The yuv colorspace, specifying range, chromaticities, and chrominance phase.</param>
        /// <param name="yuvMemSpace">The memory space where the pixel buffers reside.</param>
        /// <param name="dst">The destination image.</param>
        /// <param name="dstRect">The destination rectangle (NULL implies the whole image).</param>
        /// <param name="scale">The scale factor applied to the magnitude during transfer, typically 1, 255 or 1/255.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="tmp">The staging image.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if the operation was completed successfully.</returns>
        /// <remarks>
        /// The actual transfer region may be smaller, because the rects are clipped against the images.
        /// This is supplied for use with YUV buffers that do not have the standard structure
        /// that are expected for NvCVImage_Transfer() and NvCVImage_TransferRect.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_TransferFromYUV(
            IntPtr y,
            int yPixBytes,
            int yPitch,
            IntPtr u,
            IntPtr v,
            int uvPixBytes, 
            int uvPitch,
            NvCVImagePixelFormat yuvFormat, 
            NvCVImageComponentType yuvType,
            uint yuvColorSpace, 
            NvCVMemSpace yuvMemSpace,
            out NvCVImage dst, 
            NvCVRect2i dstRect, 
            float scale,
            IntPtr stream, 
            NvCVImage tmp);

        /// <summary>
        /// Transfer to a YUV image. RGBu8 --> YUVu8 and RGBf32 --> YUVu8 are currently available.
        /// </summary>
        /// <param name="src">The source image.</param>
        /// <param name="srcRect">The destination rectangle (NULL implies the whole image).</param>
        /// <param name="y">The pointer to pixel(0,0) of the luminance channel.</param>
        /// <param name="yPixBytes">The byte stride between y pixels horizontally.</param>
        /// <param name="yPitch">The byte stride between y pixels vertically.</param>
        /// <param name="u">The pointer to pixel(0,0) of the u (Cb) chrominance channel.</param>
        /// <param name="v">The pointer to pixel(0,0) of the v (Cr) chrominance channel.</param>
        /// <param name="uvPixBytes">The byte stride between u or v pixels horizontally.</param>
        /// <param name="uvPitch">The byte stride between u or v pixels vertically.</param>
        /// <param name="yuvFormat">The yuv format.</param>
        /// <param name="yuvType">Type of the yuv.</param>
        /// <param name="yuvColorSpace">The yuv colorspace, specifying range, chromaticities, and chrominance phase.</param>
        /// <param name="yuvMemSpace">The memory space where the pixel buffers reside.</param>
        /// <param name="scale">The scale factor applied to the magnitude during transfer, typically 1, 255 or 1/255.</param>
        /// <param name="stream">The CUDA stream.</param>
        /// <param name="tmp">The staging image.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if the operation was completed successfully.</returns>
        /// <remarks>
        /// The actual transfer region may be smaller, because the rects are clipped against the images.
        /// This is supplied for use with YUV buffers that do not have the standard structure
        /// that are expected for NvCVImage_Transfer() and NvCVImage_TransferRect.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_TransferToYUV(
            NvCVImage src, 
            NvCVRect2i srcRect,
            IntPtr y,
            int yPixBytes,
            int yPitch,
            IntPtr u,
            IntPtr v,
            int uvPixBytes, 
            int uvPitch,
            NvCVImagePixelFormat yuvFormat, 
            NvCVImageComponentType yuvType,
            uint yuvColorSpace, 
            NvCVMemSpace yuvMemSpace,
            float scale,
            IntPtr stream, 
            NvCVImage tmp);

        /// <summary>
        /// Between rendering by a graphics system and Transfer by CUDA, it is necessary to map the texture resource.
        /// There is a fair amount of overhead, so its use should be minimized.
        /// Every call to NvCVImage_MapResource() should be matched by a subsequent call to NvCVImage_UnmapResource().
        /// </summary>
        /// <param name="im">The image to be mapped.</param>
        /// <param name="stream">The stream on which the mapping is to be performed.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS is the operation was completed successfully.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_MapResource(NvCVImage im, IntPtr stream);

        /// <summary>
        /// After transfer by CUDA, the texture resource must be unmapped in order to be used by the graphics system again.
        /// There is a fair amount of overhead, so its use should be minimized.
        /// Every call to NvCVImage_UnmapResource() should correspond to a preceding call to NvCVImage_MapResource().
        /// </summary>
        /// <param name="im">The image to be mapped.</param>
        /// <param name="stream">The CUDA stream on which the mapping is to be performed.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS is the operation was completed successfully.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_UnmapResource(NvCVImage im, IntPtr stream);


        /// <summary>
        /// Composite one source image over another using the given matte. This accommodates all RGB and RGBA formats, with u8 and f32 components.
        /// </summary>
        /// <param name="fg">The foreground source image.</param>
        /// <param name="bg">The background source image.</param>
        /// <param name="mat">The matte Yu8 (or Au8) image, indicating where the src should come through.</param>
        /// <param name="dst">The destination image. This can be the same as fg or bg.</param>
        /// <param name="stream">The CUDA stream on which the composition is to be performed.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
        /// NVCV_ERR_MISMATCH if either the fg & bg & dst formats do not match, or if fg & bg & dst & mat are not
        /// in the same address space (CPU or GPU).
        /// </returns>
        /// <remarks>
        /// Though RGBA destinations are accommodated, the A channel is not updated at all.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Composite(
            NvCVImage fg, 
            NvCVImage bg,
            NvCVImage mat,
            out NvCVImage dst,
            IntPtr stream);

        /// <summary>
        /// Composite one source image over another using the given matte.
        /// This accommodates all RGB and RGBA formats, with u8 and f32 components.
        /// </summary>
        /// <param name="fg">The foreground source image.</param>
        /// <param name="fgOrg">The upper-left corner of the fg image to be composited (NULL implies (0,0)).</param>
        /// <param name="bg">The background source image.</param>
        /// <param name="bgOrg">The upper-left corner of the bg image to be composited (NULL implies (0,0)).</param>
        /// <param name="mat">
        /// The matte image, indicating where the src should come through.
        /// This determines the size of the rectangle to be composited.
        /// If this is multi-channel, the alpha channel is used as the matte.
        /// </param>
        /// <param name="mode">The composition mode: 0 (straight alpha over) or 1 (premultiplied alpha over).</param>
        /// <param name="dst">The destination image. This can be the same as fg or bg.</param>
        /// <param name="dstOrg">The upper-left corner of the dst image to be updated (NULL implies (0,0)).</param>
        /// <param name="stream">The CUDA stream on which the composition is to be performed.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
        /// NVCV_ERR_MISMATCH if either the fg & bg & dst formats do not match, or if fg & bg & dst & mat are not
        /// </returns>
        /// <remarks>
        /// If a smaller region of a matte is desired, a window can be created using
        /// NvCVImage_InitView() for chunky or NvCVImage_Init() for planar pixels.
        /// in the same address space (CPU or GPU).
        /// BUG: Though RGBA destinations are accommodated, the A channel is not updated at all.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_CompositeRect(
            NvCVImage fg,
            NvCVPoint2i fgOrg,
            NvCVImage bg,
            NvCVPoint2i bgOrg,
            NvCVImage mat, 
            uint mode,
            out NvCVImage dst,
            NvCVPoint2i dstOrg,
            IntPtr stream);

        /// <summary>
        /// Composite a source image over a constant color field using the given matte.
        /// </summary>
        /// <param name="src">The source image.</param>
        /// <param name="mat">The matte image, indicating where the src should come through.</param>
        /// <param name="bgColor">
        /// The pointer to a location holding the desired flat background color, with the same format
        /// and component ordering as the dst. This acts as a 1x1 background pixel buffer,
        /// so should reside in the same memory space (CUDA or CPU) as the other buffers.
        /// </param>
        /// <param name="dst">The destination image. May be the same as src.</param>
        /// <param name="stream">The stream.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_PIXELFORMAT if the pixel format is not accommodated.
        /// NVCV_ERR_MISMATCH if fg & mat & dst & bgColor are not in the same address space (CPU or GPU).
        /// </returns>
        /// <remarks>
        /// The bgColor must remain valid until complete; this is an important consideration especially if
        /// the buffers are on the GPU and NvCVImage_CompositeOverConstant() runs asynchronously.
        /// Though RGBA destinations are accommodated, the A channel is not updated at all.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_CompositeOverConstant(
            NvCVImage src, 
            NvCVImage mat,
            IntPtr bgColor, 
            ref NvCVImage dst,
            IntPtr stream);

        /// <summary>
        /// Flip the image vertically. No actual pixels are moved: it is just an accounting procedure.
        /// This is extremely low overhead, but requires appropriate interpretation of the pitch.
        /// Flipping twice yields the original orientation.
        /// </summary>
        /// <param name="src">The source image (NULL implies src == dst).</param>
        /// <param name="dst">The flipped image (can be the same as the src).</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if successful.
        /// NVCV_ERR_PIXELFORMAT for most planar formats.
        /// </returns>
        /// <remarks>
        /// This does not work for planar or semi-planar formats, neither RGB nor YUV.
        /// This does work for all chunky formats, including UYVY, VYUY, YUYV, YVYU.
        /// </remarks>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_FlipY(NvCVImage src, NvCVImage dst);

        /// <summary>
        /// Get the pointers for YUV, based on the format.
        /// </summary>
        /// <param name="im">The image to be deconstructed.</param>
        /// <param name="y">The place to store the pointer to y(0,0).</param>
        /// <param name="u">The place to store the pointer to u(0,0).</param>
        /// <param name="v">The place to store the pointer to v(0,0).</param>
        /// <param name="yPixBytes">The place to store the byte stride between luma samples horizontally.</param>
        /// <param name="cPixBytes">The place to store the byte stride between chroma samples horizontally.</param>
        /// <param name="yRowBytes">The place to store the byte stride between luma samples vertically.</param>
        /// <param name="cRowBytes">The place to store the byte stride between chroma samples vertically.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS - If the information was gathered successfully.
        /// NVCV_ERR_PIXELFORMAT - Otherwise.
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_GetYUVPointers(
            NvCVImage im,
            IntPtr y,
            IntPtr u, 
            IntPtr v,
            out int yPixBytes,
            out int cPixBytes, 
            out int yRowBytes, 
            out int cRowBytes);

        /// <summary>
        /// Sharpen an image.
        /// The src and dst should be the same type - conversions are not performed.
        /// This function is only implemented for NVCV_CHUNKY NVCV_U8 pixels, of format NVCV_RGB or NVCV_BGR.
        /// </summary>
        /// <param name="sharpness">The sharpness strength, calibrated so that 1 and 2 yields Adobe's Sharpen and Sharpen More.</param>
        /// <param name="src">The source image to be sharpened.</param>
        /// <param name="dst">The resultant image (may be the same as the src).</param>
        /// <param name="stream">The CUDA stream on which to perform the computations.</param>
        /// <param name="tmp">
        /// The temporary working image. This can be NULL, but may result in lower performance. 
        /// It is best if it resides on the same processor (CPU or GPU) as the destination.
        /// </param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation completed successfully.
        /// NVCV_ERR_MISMATCH if the source and destination formats are different.
        /// NVCV_ERR_PIXELFORMAT if the function has not been implemented for the chosen pixel type.
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_Sharpen(
            float sharpness, 
            NvCVImage src,
            NvCVImage dst,
            IntPtr stream,
            NvCVImage tmp);

        /// <summary>
        /// Utility to determine the D3D format from the NvCVImage format, type and layout.
        /// </summary>
        /// <param name="format">The pixel format.</param>
        /// <param name="type">The component type.</param>
        /// <param name="layout">The layout.</param>
        /// <param name="d3dFormat">The place to store the corresponding D3D format.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if successful.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_ToD3DFormat(
            NvCVImagePixelFormat format, 
            NvCVImageComponentType type, 
            NvCVLayout layout, 
            IntPtr /*DXGI_FORMAT* */ d3dFormat);
        
        /// <summary>
        /// Utility to determine the NvCVImage format, component type and layout from a D3D format.
        /// </summary>
        /// <param name="d3dFormat">The D3D format to translate.</param>
        /// <param name="format">The place to store the NvCVImage pixel format.</param>
        /// <param name="type">The place to store the NvCVImage component type.</param>
        /// <param name="layout">The place to store the NvCVImage layout.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if successful.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_FromD3DFormat(
            IntPtr /*DXGI_FORMAT*/ d3dFormat, 
            out NvCVImagePixelFormat format, 
            out NvCVImageComponentType type, 
            out NvCVLayout layout);

        /// <summary>
        /// Utility to determine the D3D color space from the NvCVImage color space.
        /// </summary>
        /// <param name="nvcvColorSpace">The color space.</param>
        /// <param name="pD3dColorSpace">The place to store the resultant D3D color space.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if successful.
        /// NVCV_ERR_PIXELFORMAT if there is no equivalent color space.
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_ToD3DColorSpace(
            uint nvcvColorSpace, 
            IntPtr /*DXGI_COLOR_SPACE_TYPE* */ pD3dColorSpace);

        /// <summary>
        /// Utility to determine the NvCVImage color space from the D3D color space.
        /// </summary>
        /// <param name="d3dColorSpace">The D3D color space.</param>
        /// <param name="pNvcvColorSpace">The place to store the resultant NvCVImage color space.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if successful.
        /// NVCV_ERR_PIXELFORMAT if there is no equivalent color space.
        /// </returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_FromD3DColorSpace(
            IntPtr /* DXGI_COLOR_SPACE_TYPE*/ d3dColorSpace, 
            uint pNvcvColorSpace);

        /// <summary>
        /// Initialize an NvCVImage from a D3D11 texture.
        /// The pixelFormat and component types with be transferred over, and a cudaGraphicsResource will be registered;
        /// the NvCVImage destructor will unregister the resource.
        /// It is necessary to call NvCVImage_MapResource() after rendering D3D and before calling  NvCVImage_Transfer(),
        /// and to call NvCVImage_UnmapResource() before rendering in D3D again.
        /// </summary>
        /// <param name="im">The image to be initialized.</param>
        /// <param name="tx">The texture to be used for initialization.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if successful.</returns>
        [DllImport(NvCVImageLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvCVImage_InitFromD3D11Texture(
            NvCVImage im, 
            object /*ID3D11Texture2D* */ tx);




    }
}
