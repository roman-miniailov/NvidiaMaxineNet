using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
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
    }
}
