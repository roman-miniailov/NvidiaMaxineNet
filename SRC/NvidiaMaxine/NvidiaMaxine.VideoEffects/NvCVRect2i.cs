using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Integer rectangle.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NvCVRect2i
    {
        /// <summary>
        /// The left edge of the rectangle.
        /// </summary>
        public int X;

        /// <summary>
        /// The top edge of the rectangle.
        /// </summary>
        public int Y;

        /// <summary>
        /// The width of the rectangle.
        /// </summary>
        public int Width;

        /// <summary>
        /// The height of the rectangle.
        /// </summary>
        public int Height;
    }
}
