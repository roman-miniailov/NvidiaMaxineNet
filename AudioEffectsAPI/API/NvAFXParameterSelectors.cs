// ***********************************************************************
// Assembly         : NvidiaMaxine.AudioEffects
// Author           : Roman Miniailov
// Created          : 12-27-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-27-2022
// ***********************************************************************
// <copyright file="NvAFXParameterSelectors.cs" company="Roman Miniailov">
//     2022-2023
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace NvidiaMaxine.AudioEffects.API
{
    /// <summary>
    /// AFX parameter selectors.
    /// </summary>
    public static class NvAFXParameterSelectors
    {
        /// <summary>
        /// Number of audio streams in I/O (unsigned int).
        /// </summary>
        public static string NVAFX_PARAM_NUM_STREAMS = "num_streams";

        /// <summary>
        /// To set if SDK should select the default GPU to run the effects in a Multi-GPU setup(unsigned int). Default value is 0. Please see user manual for details.
        /// </summary>
        public static string NVAFX_PARAM_USE_DEFAULT_GPU = "use_default_gpu";

        /// <summary>
        /// To be set to '1' if SDK user wants to create and manage own CUDA context. Other users can simply
        /// ignore this parameter. Once set to '1' this cannot be unset for that session rw param
        /// Note: NVAFX_PARAM_USE_DEFAULT_GPU and NVAFX_PARAM_USER_CUDA_CONTEXT cannot be used at the same time.
        /// </summary>
        public static string NVAFX_PARAM_USER_CUDA_CONTEXT = "user_cuda_context";

        /// <summary>
        /// To be set to '1' if SDK user wants to disable cuda graphs. Other users can simply ignore this parameter.
        /// Using Cuda Graphs helps to reduce the inference between GPU and CPU which makes operations quicker.
        /// </summary>
        public static string NVAFX_PARAM_DISABLE_CUDA_GRAPH = "disable_cuda_graph";

        /// <summary>
        /// To be set to '1' if SDK user wants to enable VAD.
        /// </summary>
        public static string NVAFX_PARAM_ENABLE_VAD = "enable_vad";

        /// <summary>
        /// Model path.
        /// </summary>
        public static string NVAFX_PARAM_MODEL_PATH = "model_path";

        /// <summary>
        /// Input Sample rate. Currently supported sample rate(s): 48000, 16000, 8000.
        /// </summary>
        public static string NVAFX_PARAM_INPUT_SAMPLE_RATE = "input_sample_rate";

        /// <summary>
        /// Output Sample rate. Currently supported sample rate(s): 48000, 16000.
        /// </summary>
        public static string NVAFX_PARAM_OUTPUT_SAMPLE_RATE = "output_sample_rate";

        /// <summary>
        /// Number of input samples per frame. This is immutable parameter.
        /// </summary>
        public static string NVAFX_PARAM_NUM_INPUT_SAMPLES_PER_FRAME = "num_input_samples_per_frame";

        /// <summary>
        /// Number of output samples per frame. This is immutable parameter.
        /// </summary>
        public static string NVAFX_PARAM_NUM_OUTPUT_SAMPLES_PER_FRAME = "num_output_samples_per_frame";

        /// <summary>
        /// Number of input audio channels.
        /// </summary>
        public static string NVAFX_PARAM_NUM_INPUT_CHANNELS = "num_input_channels";

        /// <summary>
        /// Number of output audio channels.
        /// </summary>
        public static string NVAFX_PARAM_NUM_OUTPUT_CHANNELS = "num_output_channels";

        /// <summary>
        /// Effect intensity factor.
        /// </summary>
        public static string NVAFX_PARAM_INTENSITY_RATIO = "intensity_ratio";
    }
}
