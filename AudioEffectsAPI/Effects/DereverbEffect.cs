// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-28-2022
// ***********************************************************************
// <copyright file="DereverbEffect.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

using System;
using System.Diagnostics;
using NvidiaMaxine.AudioEffects.API;

namespace NvidiaMaxine.AudioEffects.Effects
{
    /// <summary>
    /// Dereverb effect.
    /// </summary>
    public class DereverbEffect : BaseEffect
    {
        /// <summary>
        /// The intensity ratio.
        /// </summary>
        private float _intensityRatio = 1.0f;

        /// <summary>
        /// Initializes a new instance of the <see cref="DereverbEffect"/> class.
        /// </summary>
        /// <param name="modelDir">The model directory.</param>
        /// <param name="sampleRate">The sample rate.</param>
        public DereverbEffect(string modelDir, SampleRate sampleRate)
            : base(modelDir, sampleRate)
        {
            if (sampleRate == SampleRate.SR8000)
            {
                throw new ArgumentException("Dereverb effect supports 16000 Hz and 48000 Hz.");
            }

            TAG = "Dereverb";
        }

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
        /// Gets the name of the model.
        /// </summary>
        /// <returns>System.String.</returns>
        protected override string GetModelName()
        {
            if (_sampleRate == SampleRate.SR48000)
            {
                return "dereverb_48k.trtpkg";
            }
            else
            {
                return "dereverb_16k.trtpkg";
            }
        }

        /// <summary>
        /// Gets the name of the effect.
        /// </summary>
        /// <returns>System.String.</returns>
        protected override string GetEffectName()
        {
            return NvAFXEffectSelectors.NVAFX_EFFECT_DEREVERB;
        }

        /// <summary>
        /// Applies the settings.
        /// </summary>
        /// <returns>NvAFXStatus.</returns>
        protected override NvAFXStatus ApplySettings()
        {
            // Set intensity of FX
            float intensity_ratio = _intensityRatio;
            var result = NvAFXAPI.NvAFX_SetFloat(_handle, NvAFXParameterSelectors.NVAFX_PARAM_INTENSITY_RATIO, intensity_ratio);
            if (result != NvAFXStatus.NVAFX_STATUS_SUCCESS)
            {
                Debug.WriteLine($"{TAG} Error CreateEffect: NvAFX_SetFloat(Intensity Ratio: {intensity_ratio}) failed, error {result}");
            }

            return result;
        }
    }
}
