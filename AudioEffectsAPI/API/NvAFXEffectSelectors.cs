// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="NvAFXEffectSelectors.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.AudioEffects.API
{
    /// <summary>
    /// NvAFX effect selectors.
    /// </summary>
    public static class NvAFXEffectSelectors
    {
        /// <summary>
        /// Denoiser Effect.
        /// </summary>
        public static string NVAFX_EFFECT_DENOISER = "denoiser";

        /// <summary>
        /// Dereverb Effect.
        /// </summary>
        public static string NVAFX_EFFECT_DEREVERB = "dereverb";

        /// <summary>
        /// Dereverb Denoiser Effect.
        /// </summary>
        public static string NVAFX_EFFECT_DEREVERB_DENOISER = "dereverb_denoiser";

        /// <summary>
        /// Acoustic Echo Cancellation Effect.
        /// </summary>
        public static string NVAFX_EFFECT_AEC = "aec";

        /// <summary>
        /// Super-resolution Effect.
        /// </summary>
        public static string NVAFX_EFFECT_SUPERRES = "superres";

        /// <summary>
        /// The chained effect denoiser 16K + superres 16K to 48K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_DENOISER_16k_SUPERRES_16k_TO_48k = "denoiser16k_superres16kto48k";

        /// <summary>
        /// The chained effect dereverb 16K + superres 16K to 48K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_DEREVERB_16k_SUPERRES_16k_TO_48k = "dereverb16k_superres16kto48k";

        /// <summary>
        /// The chained effect dereverb + denoiser 16K + superres 16K to 48K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_DEREVERB_DENOISER_16k_SUPERRES_16k_TO_48k = "dereverb_denoiser16k_superres16kto48k";

        /// <summary>
        /// The chained effect superres 8k to 16K + denoiser 16K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DENOISER_16k = "superres8kto16k_denoiser16k";

        /// <summary>
        /// The chained effect superres 8k to 16K + dereverb 16K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_16k = "superres8kto16k_dereverb16k";

        /// <summary>
        /// The chained effect superres 8k to 16K + dereverb + denoiser 16K.
        /// </summary>
        public static string NVAFX_CHAINED_EFFECT_SUPERRES_8k_TO_16k_DEREVERB_DENOISER_16k = "superres8kto16k_dereverb_denoiser16k";

    }
}
