// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="ImageHelper.cs" company="">
//     Copyright (c) 2006-2022
// </copyright>
// <summary></summary>
// ***********************************************************************

using System.Drawing.Imaging;
using System.Drawing;
using System;
using NvidiaMaxine.VideoEffects.API;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Image helper.
    /// </summary>
    internal static class ImageHelper
    {
        /// <summary>
        /// Gets the stride for RGB24.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <returns>System.Int32.</returns>
        public static int GetStrideRGB24(int width)
        {
            int stride = ((width * 3) - 1) / 4 * 4 + 4;
            return stride;
        }

        /// <summary>
        /// Gets the stride for RGB32.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <returns>System.Int32.</returns>
        public static int GetStrideRGB32(int width)
        {
            int stride = ((width * 4) - 1) / 4 * 4 + 4;
            return stride;
        }

        /// <summary>
        /// Converts the specified source to the new pixelformat.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="px">The pixel format.</param>
        /// <returns>Bitmap.</returns>
        public static Bitmap Convert(this Bitmap source, PixelFormat px)
        {
            var target = new Bitmap(source.Width, source.Height, px);

            using (Graphics g = Graphics.FromImage(target))
            {
                g.DrawImage(source, new Rectangle(0, 0, target.Width, target.Height), new Rectangle(0, 0, source.Width, source.Height), GraphicsUnit.Pixel);
            }

            return target;
        }

        /// <summary>
        /// Converts the specified source to the new pixelformat.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="px">The pixel format.</param>
        /// <returns>Bitmap.</returns>
        public static VideoFrame Convert(this VideoFrame source, PixelFormat px)
        {
            var targetBmp = new Bitmap(source.Width, source.Height, px);
            var sourceBmp = source.ToBitmap();

            using (Graphics g = Graphics.FromImage(targetBmp))
            {
                g.DrawImage(sourceBmp, new Rectangle(0, 0, targetBmp.Width, targetBmp.Height), new Rectangle(0, 0, source.Width, source.Height), GraphicsUnit.Pixel);
            }

            var target = targetBmp.ToVideoFrame();

            sourceBmp.Dispose();
            targetBmp.Dispose();

            return target;
        }

        /// <summary>
        /// Converts the grayscale to RGB.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="size">The size.</param>
        /// <returns>IntPtr.</returns>
        public static IntPtr ConvertGrayscaleToRGB(this IntPtr source, int size)
        {
            var result = Marshal.AllocHGlobal(size * 3);

            for (int i = 0; i < size; i++)
            {
                Marshal.WriteByte(result, i * 3, Marshal.ReadByte(source, i));
                Marshal.WriteByte(result, i * 3 + 1, Marshal.ReadByte(source, i));
                Marshal.WriteByte(result, i * 3 + 2, Marshal.ReadByte(source, i));
            }

            return result;
        }

        /// <summary>
        /// Converts the grayscale to RGB.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="size">The size.</param>
        /// <returns>VideoFrame.</returns>
        public static VideoFrame ConvertGrayscaleToRGB(this VideoFrame source)
        {
            var image = new VideoFrame(source.Width, source.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);

            for (int i = 0; i < source.DataSize; i++)
            {
                Marshal.WriteByte(image.Data, i * 3, Marshal.ReadByte(source.Data, i));
                Marshal.WriteByte(image.Data, i * 3 + 1, Marshal.ReadByte(source.Data, i));
                Marshal.WriteByte(image.Data, i * 3 + 2, Marshal.ReadByte(source.Data, i));
            }

            //{
            //    image.Data[i * 3] = source.Data[i];
            //    image.Data[i * 3 + 1] = source.Data[i];
            //    image.Data[i * 3 + 2] = source.Data[i];
            //}

            return image;
        }

        /// <summary>
        /// Converts Bitmap to RAW byte array.
        /// </summary>
        /// <param name="sourceBmp">The source bitmap.</param>
        /// <param name="destArray">The destination array.</param>
        /// <param name="flip">if set to <c>true</c> flip.</param>
        public static void ToIntPtr(this Bitmap sourceBmp, IntPtr destArray, bool flip = false)
        {
            if (flip)
            {
                sourceBmp.RotateFlip(RotateFlipType.RotateNoneFlipY);
            }

            BitmapData bitmapData = new BitmapData();
            Rectangle rect = new Rectangle(0, 0, sourceBmp.Width, sourceBmp.Height);

            if (destArray == IntPtr.Zero)
            {
                throw new Exception("destArray is NULL, you must allocate memory before usage!");
            }

            if (sourceBmp.PixelFormat == PixelFormat.Format24bppRgb)
            {
                sourceBmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb, bitmapData);

                IntPtr src = bitmapData.Scan0;

                int strideSrc = bitmapData.Stride;
                int strideDest = GetStrideRGB24(sourceBmp.Width);

                for (int i = 0; i < sourceBmp.Height; i++)
                {
                    IntPtr tmpDest = new IntPtr(destArray.ToInt64() + (strideDest * i));
                    IntPtr tmpSrc = new IntPtr(src.ToInt64() + (strideSrc * i));

                    WinAPI.CopyMemory(tmpDest, tmpSrc, strideDest);
                }

                sourceBmp.UnlockBits(bitmapData);
            }
            else if (sourceBmp.PixelFormat == PixelFormat.Format32bppArgb)
            {
                sourceBmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb, bitmapData);

                IntPtr src = bitmapData.Scan0;

                int strideSrc = bitmapData.Stride;
                int strideDest = GetStrideRGB32(sourceBmp.Width);

                for (int i = 0; i < sourceBmp.Height; i++)
                {
                    IntPtr tmpDest = new IntPtr(destArray.ToInt64() + (strideDest * i));
                    IntPtr tmpSrc = new IntPtr(src.ToInt64() + (strideSrc * i));

                    WinAPI.CopyMemory(tmpDest, tmpSrc, strideDest);
                }

                sourceBmp.UnlockBits(bitmapData);
            }
        }

        /// <summary>
        /// Crops the specified source.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="cropRect">The crop rectangle.</param>
        /// <returns>Bitmap.</returns>
        public static Bitmap Crop(Bitmap source, Rectangle cropRect)
        {
            var target = new Bitmap(cropRect.Width, cropRect.Height);

            using (Graphics g = Graphics.FromImage(target))
            {
                g.DrawImage(source, new Rectangle(0, 0, target.Width, target.Height), cropRect, GraphicsUnit.Pixel);
            }

            return target;
        }

        /// <summary>
        /// Crops the specified source.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="cropRect">The crop rectangle.</param>
        /// <returns>Bitmap.</returns>
        public static VideoFrame Crop(VideoFrame source, Rectangle cropRect)
        {
            var sourceBitmap = source.ToBitmap();
            var target = new Bitmap(cropRect.Width, cropRect.Height);

            using (Graphics g = Graphics.FromImage(target))
            {
                g.DrawImage(sourceBitmap, new Rectangle(0, 0, target.Width, target.Height), cropRect, GraphicsUnit.Pixel);
            }

            sourceBitmap.Dispose();

            return target.ToVideoFrame();
        }

        /// <summary>
        /// Converts VideoFrame to Bitmap.
        /// </summary>
        /// <param name="frame">The frame.</param>
        /// <param name="horizontalFlip">Horizontal flip.</param>
        /// <returns>Bitmap.</returns>
        public static Bitmap ToBitmap(this VideoFrame frame)
        {
            PixelFormat pf;
            switch (frame.PixelFormat)
            {
                case NvCVImagePixelFormat.NVCV_RGB:
                case NvCVImagePixelFormat.NVCV_BGR:
                    pf = PixelFormat.Format24bppRgb;
                    break;
                case NvCVImagePixelFormat.NVCV_RGBA:
                case NvCVImagePixelFormat.NVCV_BGRA:
                    pf = PixelFormat.Format32bppArgb;
                    break;
                case NvCVImagePixelFormat.NVCV_Y:
                    pf = PixelFormat.Format8bppIndexed;
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            return IntPtrToBitmap(frame.Data, frame.Width, frame.Height, pf);
        }

        /// <summary>
        /// Converts Bitmap to RAW byte array.
        /// </summary>
        /// <param name="sourceBmp">The source bitmap.</param>
        /// <param name="destArray">The destination array.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="pixelFormat">The pixel format.</param>
        public static void BitmapToIntPtr(
            Bitmap sourceBmp, IntPtr destArray, int width, int height, PixelFormat pixelFormat)
        {
            if (destArray == IntPtr.Zero)
            {
                throw new Exception("destArray is NULL, you must allocate memory before usage!");
            }

            BitmapData bitmapData = new BitmapData();
            Rectangle rect = new Rectangle(0, 0, width, height);

            Bitmap bitmap;
            bool disposeBitmap;

            if (sourceBmp == null)
            {
                return;
            }

            if (width != sourceBmp.Width || height != sourceBmp.Height)
            {
                bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);

                using (Graphics g = Graphics.FromImage(bitmap))
                {
                    g.DrawImage(sourceBmp, new Rectangle(Point.Empty, bitmap.Size));
                }

                disposeBitmap = true;
            }
            else
            {
                bitmap = sourceBmp;

                disposeBitmap = false;
            }

            if (pixelFormat == PixelFormat.Format24bppRgb)
            {
                bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb, bitmapData);

                IntPtr src = bitmapData.Scan0;

                int strideSrc = bitmapData.Stride;
                int strideDest = GetStrideRGB24(bitmap.Width);

                if (strideSrc == strideDest)
                {
                    WinAPI.CopyMemory(destArray, src, strideSrc * bitmap.Height);
                }
                else
                {
                    for (int i = 0; i < bitmap.Height; i++)
                    {
                        IntPtr tmpDest = new IntPtr(destArray.ToInt64() + (strideDest * i));
                        IntPtr tmpSrc = new IntPtr(src.ToInt64() + (strideSrc * i));

                        WinAPI.CopyMemory(tmpDest, tmpSrc, strideDest);
                    }
                }

                bitmap.UnlockBits(bitmapData);
                // ReSharper disable once RedundantAssignment
                bitmapData = null;
            }
            else if (pixelFormat == PixelFormat.Format32bppArgb)
            {
                bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb, bitmapData);

                IntPtr src = bitmapData.Scan0;

                int strideSrc = bitmapData.Stride;
                int strideDest = GetStrideRGB32(bitmap.Width);

                if (strideSrc == strideDest)
                {
                    WinAPI.CopyMemory(destArray, src, strideSrc * bitmap.Height);
                }
                else
                {
                    for (int i = 0; i < bitmap.Height; i++)
                    {
                        IntPtr tmpDest = new IntPtr(destArray.ToInt64() + (strideDest * i));
                        IntPtr tmpSrc = new IntPtr(src.ToInt64() + (strideSrc * i));

                        WinAPI.CopyMemory(tmpDest, tmpSrc, strideDest);
                    }
                }

                bitmap.UnlockBits(bitmapData);
                // ReSharper disable once RedundantAssignment
                bitmapData = null;
            }

            if (disposeBitmap)
            {
                bitmap.Dispose();
            }
        }
        
        /// <summary>
        /// Converts Bitmap to VideoFrame.
        /// </summary>
        /// <param name="bitmap">The bitmap.</param>
        /// <returns>RAWImage.</returns>
        public static VideoFrame ToVideoFrame(this Bitmap bitmap)
        {
            var raw = new VideoFrame(bitmap.Width, bitmap.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);

            BitmapToIntPtr(bitmap, raw.Data, bitmap.Width, bitmap.Height, PixelFormat.Format24bppRgb);

            return raw;
        }

        /// <summary>
        /// Converts IntPtr to Bitmap.
        /// </summary>
        /// <param name="source">Source.</param>
        /// <param name="width">Width.</param>
        /// <param name="height">Height.</param>
        /// <param name="pixelFormat">Pixel format.</param>
        /// <returns>Bitmap.</returns>
        /// <exception cref="System.Exception">Unsupported pixel format.</exception>
        public static Bitmap IntPtrToBitmap(IntPtr source, int width, int height, PixelFormat pixelFormat)
        {
            Bitmap outputBitmap;
            if (pixelFormat != PixelFormat.Format24bppRgb && pixelFormat != PixelFormat.Format32bppArgb)
            {
                return null;
            }

            int stride;
            switch (pixelFormat)
            {
                case PixelFormat.Format24bppRgb:
                    stride = GetStrideRGB24(width);
                    break;
                case PixelFormat.Format32bppArgb:
                    stride = GetStrideRGB32(width);
                    break;
                default:
                    throw new Exception("Unsupported pixel format.");
            }

            int bufSize = stride * height;

            outputBitmap = new Bitmap(width, height, pixelFormat);
            BitmapData bitmapData = outputBitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, pixelFormat);

            WinAPI.CopyMemory(bitmapData.Scan0, source, bufSize);

            outputBitmap.UnlockBits(bitmapData);

            return outputBitmap;
        }

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <param name="fast">Fast but low quality resize.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(this Image image, int width, int height, bool fast = false)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                if (fast)
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighSpeed;
                    graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                    graphics.SmoothingMode = SmoothingMode.HighSpeed;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighSpeed;
                }
                else
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                }
                
                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            image.Dispose();

            return destImage;
        }

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <param name="fast">Fast but low quality resize.</param>
        /// <returns>The resized image.</returns>
        public static VideoFrame ResizeImage24(this VideoFrame image, int width, int height, bool fast = false)
        {           
            var srcBitmap = image.ToBitmap();            
            var destBitmap = ResizeImage(srcBitmap, width, height, fast);            
            var destImage = destBitmap.ToVideoFrame();

            srcBitmap.Dispose();
            destBitmap.Dispose();

            return destImage;
        }
    }
}
