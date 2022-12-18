using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.API
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

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVRect2i"/> struct.
        /// </summary>
        public NvCVRect2i()
        {
            X = 0;
            Y = 0;
            Width = 0;
            Height = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVRect2i"/> struct.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public NvCVRect2i(int x, int y, int width, int height)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }
    }
}
