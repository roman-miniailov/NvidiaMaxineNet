// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : NightVsKnight, Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-27-2022
// ***********************************************************************
// <copyright file="BufferWrapperMarshaler.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.AudioEffects.API
{
    internal sealed class BufferWrapperMarshaler : ICustomMarshaler
    {
        private static BufferWrapperMarshaler instance = new BufferWrapperMarshaler();

        static ICustomMarshaler GetInstance(string cookie) { return instance; }

        public void CleanUpManagedData(object ManagedObj)
        {
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
        }

        public int GetNativeDataSize()
        {
            return -1;
        }

        public IntPtr MarshalManagedToNative(object ManagedObj)
        {
            if (ManagedObj is BufferWrapper)
            {
                return ((BufferWrapper)ManagedObj).CopyManagedBufferToNativeBufferAndGetPointer();
            }

            throw new ArgumentException("ManagedObj must be of type BufferWrapper");
        }

        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            return null;
        }
    }
}
