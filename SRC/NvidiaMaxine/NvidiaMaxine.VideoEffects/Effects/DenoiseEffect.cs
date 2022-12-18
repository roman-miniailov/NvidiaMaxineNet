using CUDA;
using NvidiaMaxine.VideoEffects.API;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class DenoiseEffect : BaseEffect
    {
        public float Strength { get; set; } = 0;

        public DenoiseEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_DENOISING, modelsDir, sourceImage)
        {
            
        }

        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetF32(_handle, NvVFXParameterSelectors.NVVFX_STRENGTH, Strength));
        }
    }
}
