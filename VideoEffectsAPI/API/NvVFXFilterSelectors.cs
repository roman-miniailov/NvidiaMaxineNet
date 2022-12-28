using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// VFX filter selectors.
    /// </summary>
    public static class NvVFXFilterSelectors
    {
        /// <summary>
        /// Transfer.
        /// </summary>
        public static string NVVFX_FX_TRANSFER = "Transfer";

        /// <summary>
        /// Green Screen.
        /// </summary>
        public static string NVVFX_FX_GREEN_SCREEN = "GreenScreen";

        /// <summary>
        /// Background blur.
        /// </summary>
        public static string NVVFX_FX_BGBLUR = "BackgroundBlur";

        /// <summary>
        /// Artifact Reduction.
        /// </summary>
        public static string NVVFX_FX_ARTIFACT_REDUCTION = "ArtifactReduction";

        /// <summary>
        /// Super Res.
        /// </summary>
        public static string NVVFX_FX_SUPER_RES = "SuperRes";

        /// <summary>
        /// Super Res Upscale.
        /// </summary>
        public static string NVVFX_FX_SR_UPSCALE = "Upscale";

        /// <summary>
        ///  Denoising.
        /// </summary>
        public static string NVVFX_FX_DENOISING = "Denoising";
    }
}
