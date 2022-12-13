namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// MemSpace.
    /// </summary>
    public enum NvCVMemSpace
    {
        /// <summary>
        /// The buffer is stored in CPU memory.
        /// </summary>
        NVCV_CPU = 0,

        /// <summary>
        /// The buffer is stored in CUDA memory.
        /// </summary>
        NVCV_GPU = 1,

        /// <summary>
        /// The buffer is stored in CUDA memory.
        /// </summary>
        NVCV_CUDA = 1,

        /// <summary>
        /// The buffer is stored in pinned CPU memory.
        /// </summary>
        NVCV_CPU_PINNED = 2,

        /// <summary>
        ///  A CUDA array is used for storage.
        /// </summary>
        NVCV_CUDA_ARRAY = 3
    }
}
