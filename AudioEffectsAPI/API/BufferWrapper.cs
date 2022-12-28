// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : NightVsKnight, Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-27-2022
// ***********************************************************************
// <copyright file="BufferWrapper.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.AudioEffects.API
{
    public class BufferWrapper : IDisposable
    {
        private static Queue<BufferWrapper> pool = new Queue<BufferWrapper>();

        public static BufferWrapper Get(float[] buffer, int offset, int count)
        {
            BufferWrapper bufferWrapper;
            if (pool.Count == 0)
            {
                bufferWrapper = new BufferWrapper(buffer, offset, count);
            }
            else
            {
                bufferWrapper = pool.Dequeue();
                bufferWrapper.Init(buffer, offset, count);
            }

            return bufferWrapper;
        }

        public static void Return(BufferWrapper bufferWrapper)
        {
            pool.Enqueue(bufferWrapper);
        }

        public static void Clear()
        {
            while (pool.Count > 0)
            {
                var bufferWrapper = pool.Dequeue();
                bufferWrapper.Dispose();
            }
        }

        private BufferWrapper(float[] buffer, int offset, int count)
        {
            Init(buffer, offset, count);
        }

        public void Dispose()
        {
            hFinal.Free();
            hBuffers[0]?.Free();
            hBuffers[0] = null;
        }

        private float[] buffer;
        private int offset;
        private int count;

        private float[][] buffers = new float[1][];
        private GCHandle?[] hBuffers = new GCHandle?[1];
        private IntPtr[] pBuffers = new IntPtr[1];
        private GCHandle hFinal;
        private IntPtr pFinal = IntPtr.Zero;

        public void Init(float[] buffer, int offset, int count)
        {
            this.buffer = buffer;
            this.offset = offset;
            this.count = count;

            var buffer0 = buffers[0];
            if (buffer0 == null || buffer0.Length < count)
            {
                buffers[0] = new float[count];
                hBuffers[0]?.Free();
                var hBuffer = GCHandle.Alloc(buffers[0], GCHandleType.Pinned);
                hBuffers[0] = hBuffer;
                pBuffers[0] = hBuffer.AddrOfPinnedObject();
                hFinal = GCHandle.Alloc(pBuffers, GCHandleType.Pinned);
                pFinal = hFinal.AddrOfPinnedObject();
            }
        }

        internal IntPtr CopyManagedBufferToNativeBufferAndGetPointer()
        {
            Marshal.Copy(buffer, offset, pBuffers[0], count);
            return pFinal;
        }

        public void CopyNativeBufferToManagedBuffer()
        {
            Marshal.Copy(pBuffers[0], buffer, offset, count);
        }
    }
}