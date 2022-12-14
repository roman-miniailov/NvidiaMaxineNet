using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public static class NvVFXAPI
    {
        private const string NvVideoEffectsLib = "NVVideoEffects.dll";

        /// <summary>
        /// Get the SDK version.
        /// </summary>
        /// <param name="version">Pointer to an unsigned int set to (major << 24) | (minor << 16) | (build << 8) | 0.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the version was set.
        /// NVCV_ERR_PARAMETER if version was NULL.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetVersion(out uint version);

        /// <summary>
        /// Create an new instantiation of a video effect.
        /// </summary>
        /// <param name="code">The selector code for the desired video effect.</param>
        /// <param name="effect">The handle to the Video Effect instantiation.</param>
        /// <returns>NvCVStatus. NVCV_SUCCESS if the operation was successful.</returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_CreateEffect(string code, out NvVFXHandle effect);

        /// <summary>
        /// Delete a previously allocated video effect.
        /// </summary>
        /// <param name="effect">The effect a handle to the video effect to be deleted.</param>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvVFX_DestroyEffect(NvVFXHandle effect);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetU32(NvVFXHandle effect, string paramName, uint val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetS32(NvVFXHandle effect, string paramName, int val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetF32(NvVFXHandle effect, string paramName, float val);
        
        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetF64(NvVFXHandle effect, string paramName, double val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetU64(NvVFXHandle effect, string paramName, ulong val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetObject(NvVFXHandle effect, string paramName, IntPtr ptr);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetStateObjectHandleArray(
            NvVFXHandle effect, 
            string paramName,
            /* NvVFX_StateObjectHandle* */ object handle);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect parameter to configure.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetCudaStream(
            NvVFXHandle effect, 
            string paramName, 
            /*CUstream*/ object stream);


    }
}
