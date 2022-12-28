using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// Integer point.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct NvCVPoint2i
    {
        /// <summary>
        /// The horizontal coordinate.
        /// </summary>
        public int X;

        /// <summary>
        /// The vertical coordinate.
        /// </summary>
        public int Y;

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVPoint2i"/> struct.
        /// </summary>
        public NvCVPoint2i()
        {
            X = 0;
            Y = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NvCVPoint2i"/> struct.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="y">The y.</param>
        public NvCVPoint2i(int x, int y)
        {
            X = x;
            Y = y;
        }
    }
}
