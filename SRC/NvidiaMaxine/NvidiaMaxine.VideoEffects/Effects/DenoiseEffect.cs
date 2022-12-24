using NvidiaMaxine.VideoEffects.API;

#if OPENCV
using OpenCvSharp;
#endif

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class DenoiseEffect : BaseEffect
    {
        public float Strength { get; set; } = 0;

#if OPENCV
        public DenoiseEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_DENOISING, modelsDir, sourceImage)
#else
        public DenoiseEffect(string modelsDir, VideoFrame sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_DENOISING, modelsDir, sourceImage)
#endif
        {
            if (sourceImage.Width > 1920 || sourceImage.Height > 1080)
            {
                throw new System.Exception("Denoise effects supports up to 1920x1080 resolution.");
            }

            _useState = true;
        }

        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetF32(_handle, NvVFXParameterSelectors.NVVFX_STRENGTH, Strength));
        }
    }
}
