// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-22-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-24-2022
// ***********************************************************************
// <copyright file="VideoFrame.cs" company="">
//     Copyright (c) 2006-2022
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Video frame.
    /// Implements the <see cref="IDisposable" />.
    /// </summary>
    /// <seealso cref="IDisposable" />
    public partial class VideoFrame : IDisposable
    {
        /// <summary>
        /// The disposed value.
        /// </summary>
        private bool disposedValue;

        /// <summary>
        /// The is owner.
        /// </summary>
        private bool _isOwner;

        /// <summary>
        /// Gets or sets the data.
        /// </summary>
        /// <value>The data.</value>
        public IntPtr Data { get; set; }

        /// <summary>
        /// Gets or sets the size of the data.
        /// </summary>
        /// <value>The size of the data.</value>
        public long DataSize { get; set; }

        /// <summary>
        /// Gets or sets the width.
        /// </summary>
        /// <value>The width.</value>
        public int Width { get; set; }

        /// <summary>
        /// Gets or sets the height.
        /// </summary>
        /// <value>The height.</value>
        public int Height { get; set; }

        /// <summary>
        /// Gets or sets the stride.
        /// </summary>
        /// <value>The stride.</value>
        public int Stride { get; set; }

        /// <summary>
        /// Gets or sets the type of the component.
        /// </summary>
        /// <value>The type of the component.</value>
        public NvCVImageComponentType ComponentType { get; set; }

        /// <summary>
        /// Gets or sets the pixel format.
        /// </summary>
        /// <value>The pixel format.</value>
        public NvCVImagePixelFormat PixelFormat { get; set; }

        /// <summary>
        /// Gets or sets the pixel bytes.
        /// </summary>
        /// <value>The pixel bytes.</value>
        public byte PixelBytes { get; set; }

        /// <summary>
        /// Gets or sets the component bytes.
        /// </summary>
        /// <value>The component bytes.</value>
        public byte ComponentBytes { get; set; }

        /// <summary>
        /// Gets or sets the number components.
        /// </summary>
        /// <value>The number components.</value>
        public byte NumComponents { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        public VideoFrame()
        {
            _isOwner = true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="dataSize">Size of the data.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="pixelFormat">The pixel format.</param>
        /// <param name="componentType">Type of the component.</param>
        public VideoFrame(IntPtr data, long dataSize, int width, int height, int stride, NvCVImagePixelFormat pixelFormat, NvCVImageComponentType componentType)
        {
            Data = data;
            DataSize = dataSize;
            Width = width;
            Height = height;
            Stride = stride;
            PixelFormat = pixelFormat;
            ComponentType = componentType;
            _isOwner = false;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="pixelFormat">The pixel format.</param>
        /// <param name="componentType">Type of the component.</param>
        public VideoFrame(int width, int height, NvCVImagePixelFormat pixelFormat, NvCVImageComponentType componentType)
        {
            Width = width;
            Height = height;
            PixelFormat = pixelFormat;
            ComponentType = componentType;
            _isOwner = true;

            NumComponents = (byte)PixelFormat.GetChannelsCount();
            PixelBytes = (byte)ComponentType.GetPixelBytes(NumComponents);
            ComponentBytes = (byte)ComponentType.GetComponentBytes();

            Stride = GetStrideByPixelSize(Width, PixelBytes);
            DataSize = Stride * Height;
            Data = Marshal.AllocHGlobal((int)DataSize);            
        }

        /// <summary>
        /// Allocates this instance.
        /// </summary>
        public void Allocate()
        {
            if (_isOwner)
            {
                DataSize = Width * Height * PixelBytes * NumComponents;
                Data = Marshal.AllocHGlobal((int)DataSize);
            }
        }

        /// <summary>
        /// Frees this instance.
        /// </summary>
        public void Free()
        {
            if (_isOwner && Data != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Data);
                Data = IntPtr.Zero;
            }
        }


        /// <summary>
        /// Clears the data.
        /// </summary>
        public void Clear()
        {
            WinAPI.MemSet(Data, 0, (int)DataSize);
        }

        /// <summary>
        /// Copies to another image.
        /// </summary>
        /// <param name="dst">The destination.</param>
        public void CopyTo(VideoFrame dst)
        {
            if (Width != dst.Width || Height != dst.Height || PixelFormat != dst.PixelFormat || ComponentType != dst.ComponentType)
            {
                throw new ArgumentOutOfRangeException("Source and destination imsages should have the same format.");   
            }

            WinAPI.CopyMemory(dst.Data, Data, (int)DataSize);
        }

        /// <summary>
        /// Loads from file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns>VideoFrame.</returns>
        public static VideoFrame LoadFromFile(string filename)
        {
            var bmp = new Bitmap(filename);
            var sourceBmp = bmp.Convert(System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            var image = new VideoFrame(bmp.Width, bmp.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);

            BitmapData bitmapData = new BitmapData();
            Rectangle rect = new Rectangle(0, 0, sourceBmp.Width, sourceBmp.Height);

            sourceBmp.LockBits(rect, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb, bitmapData);

            IntPtr src = bitmapData.Scan0;

            int strideSrc = bitmapData.Stride;
            int strideDest = ImageHelper.GetStrideRGB24(sourceBmp.Width);

            for (int i = 0; i < sourceBmp.Height; i++)
            {
                IntPtr tmpDest = new IntPtr(image.Data.ToInt64() + (strideDest * i));
                IntPtr tmpSrc = new IntPtr(src.ToInt64() + (strideSrc * i));

                WinAPI.CopyMemory(tmpDest, tmpSrc, strideDest);
            }

            sourceBmp.UnlockBits(bitmapData);

            bmp.Dispose();
            sourceBmp.Dispose();

            return image;
        }

        /// <summary>
        /// Gets the stride by the pixel size.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="pixelSize">Size of the pixel.</param>
        /// <returns>System.Int32.</returns>
        private static int GetStrideByPixelSize(int width, byte pixelSize)
        {
            int stride = ((width * pixelSize) - 1) / 4 * 4 + 4;
            return stride;
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                Free();

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="VideoFrame"/> class.
        /// </summary>
        ~VideoFrame()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
