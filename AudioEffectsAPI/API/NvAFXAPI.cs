// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="NvAFXAPI.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Runtime.InteropServices;

namespace NvidiaMaxine.AudioEffects.API
{
    /// <summary>
    /// AFX API.
    /// </summary>
    public static class NvAFXAPI
    {
        /// <summary>
        /// Audio effects library filename.
        /// </summary>
        private const string NVAudioEffectsLib = "NVAudioEffects.dll";

        /// <summary>
        /// Get a list of audio effects supported.
        /// </summary>
        /// <param name="num_effects">The lumber of effects returned in effects array.</param>
        /// <param name="effects">A list of effects returned by the API. This list is statically allocated by the API implementation. Caller does not need to allocate.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetEffectList(out int num_effects, out string[] effects);

        /// <summary>
        /// Create a new instance of an audio effect.
        /// </summary>
        /// <param name="code">The selector code for the desired audio effect.</param>
        /// <param name="effect">A handle to the Audio Effect instantiation.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_CreateEffect(string code, out IntPtr effect);

        /// <summary>
        /// Create a new instance of an audio effect.
        /// </summary>
        /// <param name="code">The selector code for the desired chained audio effect.</param>
        /// <param name="effect">A handle to the Audio Effect instantiation.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_CreateChainedEffect(string code, IntPtr effect);

        /// <summary>
        /// Delete a previously instantiated audio effect.
        /// </summary>
        /// <param name="effect">A handle to the audio Effect to be deleted.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_DestroyEffect(IntPtr effect);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetU32(IntPtr effect, string param_name, uint val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <param name="size">THe size.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetU32List(IntPtr effect, string param_name, uint[] val, uint size);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetString(IntPtr effect, string param_name, string val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <param name="size">THe size.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetStringList(IntPtr effect, string param_name, string[] val, uint size);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetFloat(IntPtr effect, string param_name, float val);

        /// <summary>
        /// Set the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The selector of the effect parameter to configure.</param>
        /// <param name="param_name">Name of the parameter.</param>
        /// <param name="val">The value to be assigned to the selected effect parameter.</param>
        /// <param name="size">THe size.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_SetFloatList(IntPtr effect, string param_name, float[] val, uint size);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="param_name">The selector of the effect parameter to read.</param>
        /// <param name="val">Buffer in which the parameter value will be assigned.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetU32(IntPtr effect, string param_name, out uint val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="param_name">The selector of the effect parameter to read.</param>
        /// <param name="val">Buffer in which the parameter value will be assigned.</param>
        /// <param name="max_length">The length in bytes of the buffer provided.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetString(IntPtr effect, string param_name, out string val, int max_length);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="param_name">The selector of the effect parameter to read.</param>
        /// <param name="val">Buffer in which the parameter value will be assigned.</param>
        /// <param name="max_length">The length in bytes of the buffer provided.</param>
        /// <param name="size">The size.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetStringList(IntPtr effect, string param_name, out string[] val, out int max_length, int size);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="param_name">The selector of the effect parameter to read.</param>
        /// <param name="val">Buffer in which the parameter value will be assigned.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetFloat(IntPtr effect, string param_name, out float val);

        /// <summary>
        /// Get the value of the selected parameter.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="param_name">The selector of the effect parameter to read.</param>
        /// <param name="val">Buffer in which the parameter value will be assigned.</param>
        /// <param name="size">The size.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetFloatList(IntPtr effect, string param_name, out float[] val, uint size);

        /// <summary>
        /// Load the Effect based on the set params.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_Load(IntPtr effect);

        /// <summary>
        /// Get the devices supported by the model.
        /// This method must be called after setting model path.
        /// </summary>
        /// <param name="effect">The effect object handle.</param>
        /// <param name="num">The size of the input array. This value will be set by the function if call succeeds.</param>
        /// <param name="devices">
        /// Array of size num. The function will fill the array with CUDA device indices of devices supported by the model,
        /// in descending order of preference(first = most preferred device).
        /// /// </param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_GetSupportedDevices(IntPtr effect, out int num, IntPtr devices);

        /// <summary>
        /// Process the input buffer as per the effect selected. e.g. denoising.
        /// The input float data is expected to be standard 32-bit float type with values in range [-1.0, +1.0].
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="input">
        /// Input float buffer array. It points to an array of buffers where each buffer holds
        /// audio data for a single channel. Array size should be same as number of
        /// input channels expected by the effect. Also ensure sampling rate is same as
        /// expected by the Effect.
        /// For e.g. for denoiser it should be equal to the value returned by NvAFX_GetU32()
        /// returned value for NVAFX_FIXED_PARAM_DENOISER_SAMPLE_RATE parameter.
        /// </param>
        /// <param name="output">
        /// Output float buffer array. The layout is same as input. It points to an an array of
        /// buffers where buffer has audio data corresponding to that channel. The buffers have
        /// to be preallocated by caller. Size of each buffer (i.e. channel) is same as that of
        /// input. However, number of channels may differ (can be queried by calling
        /// NvAFX_GetU32() with NVAFX_PARAM_NUM_OUTPUT_CHANNELS as parameter).
        /// </param>
        /// <param name="num_input_samples">
        /// The number of samples in the input buffer. After this call returns output
        /// can be determined by calling NvAFX_GetU32() with
        /// NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME as parameter.
        /// </param>
        /// <param name="num_input_channels">
        /// The number of channels in the input buffer. The @a input should point
        /// to @ num_input_channels number of buffers for input, which can be determined by
        /// calling NvAFX_GetU32() with NVAFX_PARAM_NUM_INPUT_CHANNELS as parameter.
        /// </param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_Run(
            IntPtr effect,
            [In, Out, MarshalAs(UnmanagedType.LPArray)] IntPtr[] input,
            [In, Out, MarshalAs(UnmanagedType.LPArray)] IntPtr[] output,
            uint num_input_samples,
            uint num_input_channels);

        /// <summary>
        /// Process the input buffer as per the effect selected. e.g. denoising.
        /// The input float data is expected to be standard 32-bit float type with values in range [-1.0, +1.0].
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <param name="input">
        /// Input float buffer array. It points to an array of buffers where each buffer holds
        /// audio data for a single channel. Array size should be same as number of
        /// input channels expected by the effect. Also ensure sampling rate is same as
        /// expected by the Effect.
        /// For e.g. for denoiser it should be equal to the value returned by NvAFX_GetU32()
        /// returned value for NVAFX_FIXED_PARAM_DENOISER_SAMPLE_RATE parameter.
        /// </param>
        /// <param name="output">
        /// Output float buffer array. The layout is same as input. It points to an an array of
        /// buffers where buffer has audio data corresponding to that channel. The buffers have
        /// to be preallocated by caller. Size of each buffer (i.e. channel) is same as that of
        /// input. However, number of channels may differ (can be queried by calling
        /// NvAFX_GetU32() with NVAFX_PARAM_NUM_OUTPUT_CHANNELS as parameter).
        /// </param>
        /// <param name="num_input_samples">
        /// The number of samples in the input buffer. After this call returns output
        /// can be determined by calling NvAFX_GetU32() with
        /// NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME as parameter.
        /// </param>
        /// <param name="num_input_channels">
        /// The number of channels in the input buffer. The @a input should point
        /// to @ num_input_channels number of buffers for input, which can be determined by
        /// calling NvAFX_GetU32() with NVAFX_PARAM_NUM_INPUT_CHANNELS as parameter.
        /// </param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_Run(IntPtr effect,
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(BufferWrapperMarshaler))] BufferWrapper input,
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(BufferWrapperMarshaler))] BufferWrapper output,
            uint num_input_samples,
            uint num_input_channels);

        /// <summary>
        /// Reset effect state.
        /// Allows the state of an effect to be reset. This operation will reset the state of selected in the next NvAFX_Run call.
        /// </summary>
        /// <param name="effect">The effect handle.</param>
        /// <returns>NvAFXStatus.</returns>
        [DllImport(NVAudioEffectsLib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NvAFXStatus NvAFX_Reset(IntPtr effect);
    }
}