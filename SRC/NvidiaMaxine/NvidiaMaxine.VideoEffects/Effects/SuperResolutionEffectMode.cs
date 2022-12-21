namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// Super Resolution effect mode.
    /// </summary>
    public enum SuperResolutionEffectMode : uint
    {
        /// <summary>
        /// HQ source mode enhances less and removes more encoding artifacts and is suited for lower-quality videos.  
        /// </summary>
        HQSource,

        /// <summary>
        /// LQ source mode enhances more and is suited for higher quality lossless videos.
        /// </summary>
        LQSource
    }
}
