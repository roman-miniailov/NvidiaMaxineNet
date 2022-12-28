using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.API
{
    /// <summary>
    /// VFX parameter selectors.
    /// </summary>
    public static class NvVFXParameterSelectors
    {
        /// <summary>
        /// There may be multiple input images.
        /// </summary>
        public static string NVVFX_INPUT_IMAGE_0 = "SrcImage0";

        /// <summary>
        /// But there is usually only one input image.
        /// </summary>
        public static string NVVFX_INPUT_IMAGE = NVVFX_INPUT_IMAGE_0;

        /// <summary>
        /// Source Image 1.
        /// </summary>
        public static string NVVFX_INPUT_IMAGE_1 = "SrcImage1";

        /// <summary>
        /// There may be multiple output images.
        /// </summary>
        public static string NVVFX_OUTPUT_IMAGE_0 = "DstImage0";

        /// <summary>
        /// But there is usually only one output image.
        /// </summary>
        public static string NVVFX_OUTPUT_IMAGE = NVVFX_OUTPUT_IMAGE_0;

        /// <summary>
        /// The directory where the model may be found.
        /// </summary>
        public static string NVVFX_MODEL_DIRECTORY = "ModelDir";

        /// <summary>
        /// The CUDA stream to use.
        /// </summary>
        public static string NVVFX_CUDA_STREAM = "CudaStream";

        /// <summary>
        /// Enable CUDA graph to use.
        /// </summary>
        public static string NVVFX_CUDA_GRAPH = "CudaGraph";

        /// <summary>
        /// Get info about the effects.
        /// </summary>
        public static string NVVFX_INFO = "Info";

        /// <summary>
        /// Maximum width of the input supported.
        /// </summary>
        public static string NVVFX_MAX_INPUT_WIDTH = "MaxInputWidth";

        /// <summary>
        /// Maximum height of the input supported.
        /// </summary>
        public static string NVVFX_MAX_INPUT_HEIGHT = "MaxInputHeight";

        /// <summary>
        /// Maximum number of concurrent input streams.
        /// </summary>
        public static string NVVFX_MAX_NUMBER_STREAMS = "MaxNumberStreams";

        /// <summary>
        /// Scale factor.
        /// </summary>
        public static string NVVFX_SCALE = "Scale";

        /// <summary>
        /// Strength for different filters.
        /// </summary>
        public static string NVVFX_STRENGTH = "Strength";

        /// <summary>
        /// Number of strength levels.
        /// </summary>
        public static string NVVFX_STRENGTH_LEVELS = "StrengthLevels";

        /// <summary>
        /// Mode for different filters.
        /// </summary>
        public static string NVVFX_MODE = "Mode";

        /// <summary>
        /// Temporal mode: 0=image, 1=video.
        /// </summary>
        public static string NVVFX_TEMPORAL = "Temporal";

        /// <summary>
        /// Preferred GPU (optional).
        /// </summary>
        public static string NVVFX_GPU = "GPU";

        /// <summary>
        /// Batch Size (default 1).
        /// </summary>
        public static string NVVFX_BATCH_SIZE = "BatchSize";

        /// <summary>
        /// The preferred batching model to use (default 1).
        /// </summary>
        public static string NVVFX_MODEL_BATCH = "ModelBatch";

        /// <summary>
        /// State variable.
        /// </summary>
        public static string NVVFX_STATE = "State";

        /// <summary>
        /// Number of bytes needed to store state.
        /// </summary>
        public static string NVVFX_STATE_SIZE = "StateSize";

        /// <summary>
        /// Number of active state object handles.
        /// </summary>
        public static string NVVFX_STATE_COUNT = "NumStateObjects";
    }
}
