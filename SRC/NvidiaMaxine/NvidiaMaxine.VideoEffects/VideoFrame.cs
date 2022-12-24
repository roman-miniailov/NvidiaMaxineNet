using NvidiaMaxine.VideoEffects.API;

using System;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects
{
    public partial class VideoFrame : IDisposable
    {
        private bool disposedValue;

        private bool _isOwner;

        public IntPtr Data { get; set; }

        public long DataSize { get; set; }

        public int Width { get; set; }

        public int Height { get; set; }

        public int Stride { get; set; }

        public NvCVImageComponentType ComponentType { get; set; }

        public NvCVImagePixelFormat PixelFormat { get; set; }

        public byte PixelBytes { get; set; }
        
        public byte ComponentBytes { get; set; }
        
        public byte NumComponents { get; set; }

        public VideoFrame()
        {
            _isOwner = true;
        }

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

        public void Allocate()
        {
            if (_isOwner)
            {
                DataSize = Width * Height * PixelBytes * NumComponents;
                Data = Marshal.AllocHGlobal((int)DataSize);
            }
        }

        public void Free()
        {
            if (_isOwner && Data != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(Data);
                Data = IntPtr.Zero;
            }
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

        ~VideoFrame()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
