// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-19-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-22-2022
// ***********************************************************************
// <copyright file="VideoFrameEvent.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using OpenCvSharp;
using System;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Video frame event args.
    /// Implements the <see cref="EventArgs" />
    /// </summary>
    /// <seealso cref="EventArgs" />
    public class VideoFrameEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the frame.
        /// </summary>
        /// <value>The frame.</value>
        public Mat Frame { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoFrameEventArgs"/> class.
        /// </summary>
        /// <param name="frame">The frame.</param>
        public VideoFrameEventArgs(Mat frame)
        {
            Frame = frame;
        }
    }
}
