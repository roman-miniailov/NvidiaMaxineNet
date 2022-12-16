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
        public static extern NvCVStatus NvVFX_CreateEffect(string code, out IntPtr effect);

        /// <summary>
        /// Delete a previously allocated video effect.
        /// </summary>
        /// <param name="effect">The effect a handle to the video effect to be deleted.</param>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void NvVFX_DestroyEffect(IntPtr effect);

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
        public static extern NvCVStatus NvVFX_SetU32(IntPtr effect, string paramName, uint val);

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
        public static extern NvCVStatus NvVFX_SetS32(IntPtr effect, string paramName, int val);

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
        public static extern NvCVStatus NvVFX_SetF32(IntPtr effect, string paramName, float val);
        
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
        public static extern NvCVStatus NvVFX_SetF64(IntPtr effect, string paramName, double val);

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
        public static extern NvCVStatus NvVFX_SetU64(IntPtr effect, string paramName, ulong val);

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
        public static extern NvCVStatus NvVFX_SetObject(IntPtr effect, string paramName, IntPtr[] ptr);

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
            IntPtr effect, 
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
            IntPtr effect, 
            string paramName, 
            /*CUstream*/ object stream);

        /// <summary>
        /// Set the selected image descriptor. A shallow copy of the descriptor is made (preserving the pixel pointer), so that an ephemeral NvVFXImage_Init()
        /// wrapper may be used in the call to NvVFX_SetImage() if desired, without having to preserve it for the lifetime
        /// of the effect. The effect does not take ownership of the pixel buffer.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect image to configure.</param>
        /// <param name="im">Pointer to the image descriptor to be used for the selected effect image. NULL clears the selected internal image descriptor.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetImage(IntPtr effect, string paramName, NvCVImage im);

        /// <summary>
        /// Set the value of the selected string, by making a copy in the effect handle.
        /// </summary>
        /// <param name="effect">The effect to configure.</param>
        /// <param name="paramName">The selector of the effect string to configure.</param>
        /// <param name="str">The value to be assigned to the selected effect string. NULL clears the selected string.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_SetString(
            IntPtr effect,
            [MarshalAs(UnmanagedType.LPStr)] string paramName, 
            [MarshalAs(UnmanagedType.LPStr)] string str);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetU32(IntPtr effect, string paramName, out uint val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetS32(IntPtr effect, string paramName, out int val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetF32(IntPtr effect, string paramName, out float val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetF64(IntPtr effect, string paramName, out double val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetU64(IntPtr effect, string paramName, out ulong val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetObject(IntPtr effect, string paramName, IntPtr ptr);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect parameter to retrieve.</param>
        /// <param name="val">The place to store the retrieved parameter.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetCudaStream(IntPtr effect, string paramName, out /*CUstream*/ object stream);

        /// <summary>
        /// Get a copy of the selected image descriptor. 
        /// If GetImage() is called before SetImage(), the returned descriptor will be filled with zeros.
        /// Otherwise, the values will be identical to that in the previous SetImage() call,
        /// with the exception of deletePtr, deleteProc and bufferBytes, which will be 0.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect image to retrieve.</param>
        /// <param name="im">The place to store the selected image descriptor. A pointer to an empty NvCVImage 
        /// (deletePtr==NULL) should be supplied to avoid memory leaks.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetImage(IntPtr effect, string paramName, NvCVImage im);

        /// <summary>
        /// Get the specified string. If GetString() is called before SetString(), the returned string will be empty.
        /// Otherwise, the string will be identical to that in the previous SetString() call,
        /// though it will be stored in a different location.
        /// </summary>
        /// <param name="effect">The effect to be queried.</param>
        /// <param name="paramName">The selector of the effect string to retrieve.</param>
        /// <param name="str">The place to store a pointer to the selected string.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// NVCV_ERR_SELECTOR if the chosen effect does not understand the specified selector and data type.
        /// NVCV_ERR_PARAMETER if the value was out of range.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_GetString(
            IntPtr effect, 
            string paramName, 
            out string str);

        /// <summary>
        /// Run the selected effect.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <param name="async">Run the effect asynchronously if nonzero; otherwise run synchronously.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_Run(IntPtr effect, int async);

        /// <summary>
        /// Load the model based on the set params.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_Load(IntPtr effect);

        /// <summary>
        /// Wrapper for cudaStreamCreate(), if it is desired to avoid linking with the cuda lib.
        /// </summary>
        /// <param name="stream">A place to store the newly allocated stream.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful,
        /// NVCV_ERR_CUDA_VALUE if not.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_CudaStreamCreate(/*CUstream*/ object stream);

        /// <summary>
        /// Wrapper for cudaStreamDestroy(), if it is desired to avoid linking with the cuda lib.
        /// </summary>
        /// <param name="stream">The stream to destroy.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful,
        /// NVCV_ERR_CUDA_VALUE if not.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_CudaStreamDestroy(/*CUstream*/ object stream);

        /// <summary>
        /// Allocate the state object handle for a feature.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <param name="handle">The handle to the state object.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_AllocateState(IntPtr effect, out NvVFXStateObjectHandle handle);

        /// <summary>
        /// Deallocate the state object handle for stateful feature.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <param name="handle">The handle to the state object.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_DeallocateState(IntPtr effect, NvVFXStateObjectHandle handle);

        /// <summary>
        /// Reset the state object handle for stateful feature.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <param name="handle">The handle to the state object.</param>
        /// <returns>
        /// NvCVStatus.
        /// NVCV_SUCCESS if the operation was successful.
        /// NVCV_ERR_EFFECT if an invalid effect handle was supplied.
        /// </returns>
        [DllImport(NvVideoEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvCVStatus NvVFX_ResetState(IntPtr effect, NvVFXStateObjectHandle handle);

        
    }
}
