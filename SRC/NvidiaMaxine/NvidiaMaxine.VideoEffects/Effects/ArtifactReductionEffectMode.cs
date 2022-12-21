using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Artifact reduction effect mode.
    /// </summary>
    public enum ArtifactReductionEffectMode : uint
    {
        /// <summary>
        /// Mode 0 removes lesser artifacts, preserves low gradient information better, and is suited for higher bitrate videos.
        /// </summary>
        HighBitrate,

        /// <summary>
        /// Mode 1 is better suited for lower bitrate videos.
        /// </summary>
        LowBitrate
    }
}
