namespace NvidiaMaxine.VideoEffects.Image
{
    /// <summary>
    /// The data type used to represent each component of an image.
    /// </summary>
    public enum ComponentType
    {
        /// <summary>
        /// Unknown type of component.
        /// </summary>
        NVCV_TYPE_UNKNOWN = 0,

        /// <summary>
        /// Unsigned 8-bit integer.
        /// </summary>
        NVCV_U8 = 1,      

        /// <summary>
        /// Unsigned 16-bit integer.
        /// </summary>
        NVCV_U16 = 2,       

        /// <summary>
        /// Signed 16-bit integer.
        /// </summary>
        NVCV_S16 = 3,       

        /// <summary>
        /// 16-bit floating-point.
        /// </summary>
        NVCV_F16 = 4,       

        /// <summary>
        /// Unsigned 32-bit integer.
        /// </summary>
        NVCV_U32 = 5,       

        /// <summary>
        /// Signed 32-bit integer.
        /// </summary>
        NVCV_S32 = 6,       

        /// <summary>
        /// 32-bit floating-point (float).
        /// </summary>
        NVCV_F32 = 7,       

        /// <summary>
        /// Unsigned 64-bit integer.
        /// </summary>
        NVCV_U64 = 8,       

        /// <summary>
        /// Signed 64-bit integer.
        /// </summary>
        NVCV_S64 = 9,       

        /// <summary>
        /// 64-bit floating-point (double).
        /// </summary>
        NVCV_F64 = 10,      
    }
}
