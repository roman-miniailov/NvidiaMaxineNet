using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    /// <summary>
    /// Status codes returned from APIs.
    /// </summary>
    public enum NvCVStatus
    {
        /// <summary>
        /// The procedure returned successfully.
        /// </summary>
        NVCV_SUCCESS = 0,

        /// <summary>
        /// An otherwise unspecified error has occurred.
        /// </summary>
        NVCV_ERR_GENERAL = -1,

        /// <summary>
        /// The requested feature is not yet implemented.
        /// </summary>
        NVCV_ERR_UNIMPLEMENTED = -2,

        /// <summary>
        /// There is not enough memory for the requested operation.
        /// </summary>
        NVCV_ERR_MEMORY = -3,

        /// <summary>
        /// An invalid effect handle has been supplied.
        /// </summary>
        NVCV_ERR_EFFECT = -4,

        /// <summary>
        /// The given parameter selector is not valid in this effect filter.
        /// </summary>
        NVCV_ERR_SELECTOR = -5,

        /// <summary>
        /// An image buffer has not been specified.
        /// </summary>
        NVCV_ERR_BUFFER = -6,

        /// <summary>
        /// An invalid parameter value has been supplied for this effect+selector.
        /// </summary>
        NVCV_ERR_PARAMETER = -7,

        /// <summary>
        /// Some parameters are not appropriately matched.
        /// </summary>
        NVCV_ERR_MISMATCH = -8,

        /// <summary>
        /// The specified pixel format is not accommodated.
        /// </summary>
        NVCV_ERR_PIXELFORMAT = -9,

        /// <summary>
        /// Error while loading the TRT model.
        /// </summary>
        NVCV_ERR_MODEL = -10,

        /// <summary>
        /// Error loading the dynamic library.
        /// </summary>
        NVCV_ERR_LIBRARY = -11,

        /// <summary>
        /// The effect has not been properly initialized.
        /// </summary>
        NVCV_ERR_INITIALIZATION = -12,

        /// <summary>
        /// The file could not be found.
        /// </summary>
        NVCV_ERR_FILE = -13,

        /// <summary>
        /// The requested feature was not found.
        /// </summary>
        NVCV_ERR_FEATURENOTFOUND = -14,

        /// <summary>
        /// A required parameter was not set.
        /// </summary>
        NVCV_ERR_MISSINGINPUT = -15,

        /// <summary>
        /// The specified image resolution is not supported.
        /// </summary>
        NVCV_ERR_RESOLUTION = -16,

        /// <summary>
        /// The GPU is not supported.
        /// </summary>
        NVCV_ERR_UNSUPPORTEDGPU = -17,

        /// <summary>
        /// The current GPU is not the one selected.
        /// </summary>
        NVCV_ERR_WRONGGPU = -18,

        /// <summary>
        ///  The currently installed graphics driver is not supported.
        /// </summary>
        NVCV_ERR_UNSUPPORTEDDRIVER = -19,

        /// <summary>
        /// There is no model with dependencies that match this system.
        /// </summary>
        NVCV_ERR_MODELDEPENDENCIES = -20,

        /// <summary>
        /// There has been a parsing or syntax error while reading a file.
        /// </summary>
        NVCV_ERR_PARSE = -21,

        /// <summary>
        /// The specified model does not exist and has been substituted.
        /// </summary>
        NVCV_ERR_MODELSUBSTITUTION = -22,

        /// <summary>
        /// An error occurred while reading a file.
        /// </summary>
        NVCV_ERR_READ = -23,

        /// <summary>
        /// An error occurred while writing a file.
        /// </summary>
        NVCV_ERR_WRITE = -24,

        /// <summary>
        /// The selected parameter is read-only.
        /// </summary>
        NVCV_ERR_PARAMREADONLY = -25,

        /// <summary>
        /// TensorRT enqueue failed.
        /// </summary>
        NVCV_ERR_TRT_ENQUEUE = -26,

        /// <summary>
        /// Unexpected TensorRT bindings.
        /// </summary>
        NVCV_ERR_TRT_BINDINGS = -27,

        /// <summary>
        ///  An error occurred while creating a TensorRT context.
        /// </summary>
        NVCV_ERR_TRT_CONTEXT = -28,

        /// <summary>
        /// There was a problem creating the inference engine.
        /// </summary>
        NVCV_ERR_TRT_INFER = -29,

        /// <summary>
        /// There was a problem deserializing the inference runtime engine.
        /// </summary>
        NVCV_ERR_TRT_ENGINE = -30,

        /// <summary>
        /// An error has occurred in the NPP library.
        /// </summary>
        NVCV_ERR_NPP = -31,

        /// <summary>
        /// No suitable model exists for the specified parameter configuration.
        /// </summary>
        NVCV_ERR_CONFIG = -32,

        /// <summary>
        /// A supplied parameter or buffer is not large enough.
        /// </summary>
        NVCV_ERR_TOOSMALL = -33,

        /// <summary>
        /// A supplied parameter is too big.
        /// </summary>
        NVCV_ERR_TOOBIG = -34,

        /// <summary>
        /// A supplied parameter is not the expected size.
        /// </summary>
        NVCV_ERR_WRONGSIZE = -35,

        /// <summary>
        /// The specified object was not found.
        /// </summary>
        NVCV_ERR_OBJECTNOTFOUND = -36,

        /// <summary>
        /// A mathematical singularity has been encountered.
        /// </summary>
        NVCV_ERR_SINGULAR = -37,

        /// <summary>
        ///  Nothing was rendered in the specified region.
        /// </summary>
        NVCV_ERR_NOTHINGRENDERED = -38,

        /// <summary>
        /// An OpenGL error has occurred.
        /// </summary>
        NVCV_ERR_OPENGL = -98,

        /// <summary>
        /// A Direct3D error has occurred.
        /// </summary>
        NVCV_ERR_DIRECT3D = -99,

        /// <summary>
        /// CUDA errors are offset from this value.
        /// </summary>
        NVCV_ERR_CUDA_BASE = -100,

        /// <summary>
        /// A CUDA parameter is not within the acceptable range.
        /// </summary>
        NVCV_ERR_CUDA_VALUE = -101,

        /// <summary>
        /// There is not enough CUDA memory for the requested operation.
        /// </summary>
        NVCV_ERR_CUDA_MEMORY = -102,

        /// <summary>
        /// A CUDA pitch is not within the acceptable range.
        /// </summary>
        NVCV_ERR_CUDA_PITCH = -112,

        /// <summary>
        /// The CUDA driver and runtime could not be initialized.
        /// </summary>
        NVCV_ERR_CUDA_INIT = -127,

        /// <summary>
        /// The CUDA kernel launch has failed.
        /// </summary>
        NVCV_ERR_CUDA_LAUNCH = -819,

        /// <summary>
        /// No suitable kernel image is available for the device.
        /// </summary>
        NVCV_ERR_CUDA_KERNEL = -309,

        /// <summary>
        /// The installed NVIDIA CUDA driver is older than the CUDA runtime library.
        /// </summary>
        NVCV_ERR_CUDA_DRIVER = -135,

        /// <summary>
        /// The CUDA operation is not supported on the current system or device.
        /// </summary>
        NVCV_ERR_CUDA_UNSUPPORTED = -901,

        /// <summary>
        /// CUDA tried to load or store on an invalid memory address.
        /// </summary>
        NVCV_ERR_CUDA_ILLEGAL_ADDRESS = -800,

        /// <summary>
        ///  An otherwise unspecified CUDA error has been reported.
        /// </summary>
        NVCV_ERR_CUDA = -1099,
    }
}
