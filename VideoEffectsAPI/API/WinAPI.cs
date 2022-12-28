// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="WinAPI.cs" company="">
//     Copyright (c) 2006-2022
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// Class WinAPI.
    /// </summary>
    internal static class WinAPI
    {
        /// <summary>
        /// Memories the set.
        /// </summary>
        /// <param name="dest">The dest.</param>
        /// <param name="c">The c.</param>
        /// <param name="byteCount">The byte count.</param>
        /// <returns>IntPtr.</returns>
        [DllImport("msvcrt.dll", EntryPoint = "memset", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern IntPtr MemSet(IntPtr dest, int c, int byteCount);

        /// <summary>
        /// Copies the memory.
        /// </summary>
        /// <param name="destination">The destination.</param>
        /// <param name="source">The source.</param>
        /// <param name="length">The length.</param>
        [DllImport("msvcrt.dll", EntryPoint = "memcpy", CallingConvention = CallingConvention.Cdecl, SetLastError = false)]
        public static extern void CopyMemory(IntPtr destination, IntPtr source, int length);
    }
}