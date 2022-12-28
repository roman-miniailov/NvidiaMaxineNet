// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="NvAFXStatus.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.AudioEffects.API
{
    /// <summary>
    /// NvAFX API return values.
    /// </summary>
    public enum NvAFXStatus
    {
        /// <summary>
        /// Success.
        /// </summary>
        NVAFX_STATUS_SUCCESS = 0,

        /// <summary>
        /// Failure.
        /// </summary>
        NVAFX_STATUS_FAILED = 1,

        /// <summary>
        /// Handle invalid.
        /// </summary>
        NVAFX_STATUS_INVALID_HANDLE = 2,

        /// <summary>
        /// Parameter value invalid.
        /// </summary>
        NVAFX_STATUS_INVALID_PARAM = 3,

        /// <summary>
        /// Parameter value immutable.
        /// </summary>
        NVAFX_STATUS_IMMUTABLE_PARAM = 4,

        /// <summary>
        /// Insufficient data to process.
        /// </summary>
        NVAFX_STATUS_INSUFFICIENT_DATA = 5,

        /// <summary>
        /// Effect not supported.
        /// </summary>
        NVAFX_STATUS_EFFECT_NOT_AVAILABLE = 6,

        /// <summary>
        /// Given buffer length too small to hold requested data.
        /// </summary>
        NVAFX_STATUS_OUTPUT_BUFFER_TOO_SMALL = 7,

        /// <summary>
        /// Model file could not be loaded.
        /// </summary>
        NVAFX_STATUS_MODEL_LOAD_FAILED = 8,

        /// <summary>
        /// (32 bit SDK only) COM server was not registered, please see user manual for details.
        /// </summary>
        NVAFX_STATUS_32_SERVER_NOT_REGISTERED = 9,

        /// <summary>
        /// (32 bit SDK only) COM operation failed.
        /// </summary>
        NVAFX_STATUS_32_COM_ERROR = 10,

        /// <summary>
        /// The selected GPU is not supported. The SDK requires Turing and above GPU with Tensor cores.
        /// </summary>
        NVAFX_STATUS_GPU_UNSUPPORTED = 11,
    }
}
