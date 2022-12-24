namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// The data type used to represent each component of an image.
    /// </summary>
    public enum NvCVImageComponentType : int
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

    public static class NvCVImageComponentTypeExtensions
    {
        public static int GetComponentBytes(this NvCVImageComponentType componentType)
        {
            switch (componentType)
            {
                case NvCVImageComponentType.NVCV_U8:
                    return 1;
                case NvCVImageComponentType.NVCV_U16:
                case NvCVImageComponentType.NVCV_S16:
                case NvCVImageComponentType.NVCV_F16:
                    return 2;
                case NvCVImageComponentType.NVCV_U32:
                case NvCVImageComponentType.NVCV_S32:
                case NvCVImageComponentType.NVCV_F32:
                    return 4;
                case NvCVImageComponentType.NVCV_U64:
                case NvCVImageComponentType.NVCV_S64:
                case NvCVImageComponentType.NVCV_F64:
                    return 8;
                default:
                    return 0;
            }
        }

        public static int GetPixelBytes(this NvCVImageComponentType componentType, int numComponents)
        {
            return componentType.GetComponentBytes() * numComponents;
        }        
    }
}
