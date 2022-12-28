﻿// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov, NightVsKnight
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-28-2022
// ***********************************************************************
// <copyright file="DenoiserEffect.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Diagnostics;
using System.IO;
using NvidiaMaxine.AudioEffects.API;

namespace NvidiaMaxine.AudioEffects
{
    /// <summary>
    /// Denoiser effect.
    /// </summary>
    public class DenoiserEffect : IDisposable
    {
        /// <summary>
        /// The tag.
        /// </summary>
        private const string TAG = "NvAfxDenoiser";

        /// <summary>
        /// The model dir.
        /// </summary>
        private readonly string _modelDir;

        /// <summary>
        /// The intensity ratio.
        /// </summary>
        private float _intensityRatio = 1.0f;

        /// <summary>
        /// The disposed value.
        /// </summary>
        private bool disposedValue;

        /// <summary>
        /// The nvafx sample rate.
        /// </summary>
        public static uint NVAFX_SAMPLE_RATE = 48000;

        /// <summary>
        /// The nvafx frame size.
        /// </summary>
        public static uint NVAFX_FRAME_SIZE = 480; // the sdk does not explicitly set this as a constant though it relies on it

        /// <summary>
        /// The nvafx number channels.
        /// </summary>
        public static uint NVAFX_NUM_CHANNELS = 1;

        /// <summary>
        /// The handle.
        /// </summary>
        private IntPtr _handle;

        /// <summary>
        /// Gets a value indicating whether this instance is enabled.
        /// </summary>
        /// <value><c>true</c> if this instance is enabled; otherwise, <c>false</c>.</value>
        public bool IsEnabled { get { return _handle != IntPtr.Zero; } }

        /// <summary>
        /// Gets the number channels.
        /// </summary>
        /// <value>The number channels.</value>
        public int NumChannels { get { return (int)NVAFX_NUM_CHANNELS; } }

        /// <summary>
        /// Gets the number samples per frame.
        /// </summary>
        /// <value>The number samples per frame.</value>
        public int NumSamplesPerFrame { get { return (int)NVAFX_FRAME_SIZE; } }

        /// <summary>
        /// Gets the sample rate.
        /// </summary>
        /// <value>The sample rate.</value>
        public int SampleRate { get { return (int)NVAFX_SAMPLE_RATE; } }

        /// <summary>
        /// Gets the intensity ratio.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns><c>true</c> if successful, <c>false</c> otherwise.</returns>
        public bool GetIntensityRatio(out float value)
        {
            value = 0;

            if (IsEnabled)
            {
                NvAFXStatus result = NvAFXAPI.NvAFX_GetFloat(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INTENSITY_RATIO, out _intensityRatio);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetFloat(Intensity Ratio) failed, error {result}");
                    return false;
                }
            }

            value = _intensityRatio;
            return true;
        }

        /// <summary>
        /// Sets the intensity ratio.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns><c>true</c> if successful, <c>false</c> otherwise.</returns>
        public bool SetIntensityRatio(float value)
        {
            _intensityRatio = value;
            if (IsEnabled)
            {
                NvAFXStatus result = NvAFXAPI.NvAFX_SetFloat(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INTENSITY_RATIO, _intensityRatio);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_SetFloat(Intensity Ratio: {_intensityRatio}) failed, error {result}");
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="DenoiserEffect"/> class.
        /// </summary>
        /// <param name="modelDir">The model dir.</param>
        public DenoiserEffect(string modelDir)
        {
            _modelDir = modelDir;
        }

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        /// <returns><c>true</c> if successful, <c>false</c> otherwise.</returns>
        public bool Init()
        {
            Debug.WriteLine($"{TAG} Info CreateEffect()");

            if (IsEnabled)
            {
                return true;
            }

            var result = NvAFXStatus.NVAFX_STATUS_FAILED;

            try
            {
                result = NvAFXAPI.NvAFX_CreateEffect(NvAFXEffectSelectors.NVAFX_EFFECT_DENOISER, out _handle);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_CreateEffect failed, error {result}");
                    goto FAILURE;
                }

                // Set AI models path
                string modelPath = Path.Combine(_modelDir, "denoiser_48k.trtpkg");
                result = NvAFXAPI.NvAFX_SetString(_handle, NvAFXParameterSelectors.NVAFX_PARAM_MODEL_PATH, modelPath);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_SetString(Model Path: {modelPath}) failed, error {result}");
                    goto FAILURE;
                }

                // Set sample rate of FX
                uint sampleRate = (uint)SampleRate;
                result = NvAFXAPI.NvAFX_SetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INPUT_SAMPLE_RATE, sampleRate);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_SetU32(Sample Rate: {sampleRate}) failed, error {result}");
                    goto FAILURE;
                }

                // Set intensity of FX
                float intensity_ratio = _intensityRatio;
                result = NvAFXAPI.NvAFX_SetFloat(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INTENSITY_RATIO, intensity_ratio);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_SetFloat(Intensity Ratio: {intensity_ratio}) failed, error {result}");
                    goto FAILURE;
                }

                // Load FX
                result = NvAFXAPI.NvAFX_Load(_handle);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_Load() failed, error {result}");
                    goto FAILURE;
                }

                Debug.WriteLine($"{TAG} Info CreateEffect: NvAFX_Load() success!");

                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INPUT_SAMPLE_RATE, out sampleRate);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (sampleRate != NVAFX_SAMPLE_RATE)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The input sample rate {sampleRate} is not the expected {NVAFX_SAMPLE_RATE}.");
                    goto FAILURE;
                }

                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_OUTPUT_SAMPLE_RATE, out sampleRate);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (sampleRate != NVAFX_SAMPLE_RATE)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The output sample rate {sampleRate} is not the expected {NVAFX_SAMPLE_RATE}.");
                    goto FAILURE;
                }

                uint numChannels;
                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_NUM_INPUT_CHANNELS, out numChannels);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (numChannels != NVAFX_NUM_CHANNELS)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The input number of channels {numChannels} is not the expected {NVAFX_NUM_CHANNELS}.");
                    goto FAILURE;
                }

                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_NUM_OUTPUT_CHANNELS, out numChannels);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (numChannels != NVAFX_NUM_CHANNELS)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The output number of channels {numChannels} is not the expected {NVAFX_NUM_CHANNELS}.");
                    goto FAILURE;
                }

                uint numSamplesPerFrame;
                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME, out numSamplesPerFrame);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (numSamplesPerFrame != NVAFX_FRAME_SIZE)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The input samples per frame {numSamplesPerFrame} is not the expected {NVAFX_FRAME_SIZE} (= 10 ms).");
                    goto FAILURE;
                }

                result = NvAFXAPI.NvAFX_GetU32(_handle, NvAFXParameterSelectors.NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME, out numSamplesPerFrame);
                if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                {
                    Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_GetU32() failed, error {result}");
                    goto FAILURE;
                }

                if (numSamplesPerFrame != NVAFX_FRAME_SIZE)
                {
                    result = NvAFXStatus.NVAFX_STATUS_FAILED;
                    Debug.WriteLine($"{TAG} Error CreateEffect: The output samples per frame {numSamplesPerFrame} is not the expected {NVAFX_FRAME_SIZE} (= 10 ms).");
                    goto FAILURE;
                }

                Debug.WriteLine($"{TAG} Info CreateEffect: NvAFX Initialized Successfully!");
            }
            catch (Exception e)
            {
                Debug.WriteLine($"{TAG} Info CreateEffect: NvAFX Initialization Failed, error {e}!");
                if (!(e is DllNotFoundException || e is BadImageFormatException || e is EntryPointNotFoundException))
                {
                    throw e;
                }
            }

        FAILURE:
            if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
            {
                DestroyEffect();
            }

            return IsEnabled;
        }

        /// <summary>
        /// Destroys the effect.
        /// </summary>
        public void DestroyEffect()
        {
            if (_handle != IntPtr.Zero)
            {
                Debug.WriteLine($"{TAG} Info DestroyEffect()");
                try
                {
                    var result = NvAFXAPI.NvAFX_DestroyEffect(_handle);
                    if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                    {
                        Debug.WriteLine($"{TAG} Error DestroyEffect: NvAFX_DestroyEffect failed, error {result}");
                    }
                }
                catch (Exception e)
                {
                    Debug.WriteLine($"{TAG} Error DestroyEffect: NvAFX_DestroyEffect Failed, error {e}!");
                    if (!(e is DllNotFoundException || e is BadImageFormatException || e is EntryPointNotFoundException))
                    {
                        throw e;
                    }
                }

                _handle = IntPtr.Zero;
                Debug.WriteLine($"{TAG} Info DestroyEffect: NvAFX Uninitialized");
            }
        }

        /// <summary>
        /// Processes the specified buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="count">The count.</param>
        /// <returns><c>true</c> if successful, <c>false</c> otherwise.</returns>
        public bool Process(float[] buffer, int offset, int count)
        {
            try
            {
                BufferWrapper bufferWrapper;
                NvAFXStatus result;
                int end = offset + count;
                while (offset < end)
                {
                    bufferWrapper = BufferWrapper.Get(buffer, offset, NumSamplesPerFrame);
                    result = NvAFXAPI.NvAFX_Run(_handle, bufferWrapper, bufferWrapper, (uint)NumSamplesPerFrame, (uint)NumChannels);
                    if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
                    {
                        Debug.WriteLine($"{TAG} Error Run: NvAFX_Run failed, error {result}");
                        return false;
                    }

                    bufferWrapper.CopyNativeBufferToManagedBuffer();
                    BufferWrapper.Return(bufferWrapper);
                    offset += NumSamplesPerFrame;
                }

                return true;
            }
            catch (Exception e)
            {
                Debug.WriteLine($"{TAG} Error Run: Exception {e}");
                return false;
            }
        }

        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        /// <param name="disposing"><c>true</c> to release both managed and unmanaged resources; <c>false</c> to release only unmanaged resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                }

                DestroyEffect();

                disposedValue = true;
            }
        }

        /// <summary>
        /// Finalizes an instance of the <see cref="DenoiserEffect"/> class.
        /// </summary>
        ~DenoiserEffect()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
