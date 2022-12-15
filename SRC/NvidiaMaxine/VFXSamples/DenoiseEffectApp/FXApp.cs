using DenoiseEffectApp;
using NvidiaMaxine.VideoEffects;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenoiseEffectApp
{
    internal class FXApp
    {
        enum Err
        {
            errQuit = +1,                         // Application errors
            errFlag = +2,
            errRead = +3,
            errWrite = +4,
            errNone = NvCVStatus.NVCV_SUCCESS,               // Video Effects SDK errors
            errGeneral = NvCVStatus.NVCV_ERR_GENERAL,
            errUnimplemented = NvCVStatus.NVCV_ERR_UNIMPLEMENTED,
            errMemory = NvCVStatus.NVCV_ERR_MEMORY,
            errEffect = NvCVStatus.NVCV_ERR_EFFECT,
            errSelector = NvCVStatus.NVCV_ERR_SELECTOR,
            errBuffer = NvCVStatus.NVCV_ERR_BUFFER,
            errParameter = NvCVStatus.NVCV_ERR_PARAMETER,
            errMismatch = NvCVStatus.NVCV_ERR_MISMATCH,
            errPixelFormat = NvCVStatus.NVCV_ERR_PIXELFORMAT,
            errModel = NvCVStatus.NVCV_ERR_MODEL,
            errLibrary = NvCVStatus.NVCV_ERR_LIBRARY,
            errInitialization = NvCVStatus.NVCV_ERR_INITIALIZATION,
            errFileNotFound = NvCVStatus.NVCV_ERR_FILE,
            errFeatureNotFound = NvCVStatus.NVCV_ERR_FEATURENOTFOUND,
            errMissingInput = NvCVStatus.NVCV_ERR_MISSINGINPUT,
            errResolution = NvCVStatus.NVCV_ERR_RESOLUTION,
            errUnsupportedGPU = NvCVStatus.NVCV_ERR_UNSUPPORTEDGPU,
            errWrongGPU = NvCVStatus.NVCV_ERR_WRONGGPU,
            errUnsupportedDriver = NvCVStatus.NVCV_ERR_UNSUPPORTEDDRIVER,
            errCudaMemory = NvCVStatus.NVCV_ERR_CUDA_MEMORY,       // CUDA errors
            errCudaValue = NvCVStatus.NVCV_ERR_CUDA_VALUE,
            errCudaPitch = NvCVStatus.NVCV_ERR_CUDA_PITCH,
            errCudaInit = NvCVStatus.NVCV_ERR_CUDA_INIT,
            errCudaLaunch = NvCVStatus.NVCV_ERR_CUDA_LAUNCH,
            errCudaKernel = NvCVStatus.NVCV_ERR_CUDA_KERNEL,
            errCudaDriver = NvCVStatus.NVCV_ERR_CUDA_DRIVER,
            errCudaUnsupported = NvCVStatus.NVCV_ERR_CUDA_UNSUPPORTED,
            errCudaIllegalAddress = NvCVStatus.NVCV_ERR_CUDA_ILLEGAL_ADDRESS,
            errCuda = NvCVStatus.NVCV_ERR_CUDA,
        };


        NvVFXHandle _eff;
        Mat _srcImg;
        Mat _dstImg;
        NvCVImage _srcGpuBuf;
        NvCVImage _dstGpuBuf;
        NvCVImage _srcVFX;
        NvCVImage _dstVFX;
        NvCVImage _tmpVFX;  // We use the same temporary buffer for source and dst, since it auto-shapes as needed
        bool _show;
        bool _inited;
        bool _showFPS;
        bool _progress;
        bool _enableEffect;
        bool _drawVisualization;
        string _effectName;
        float _framePeriod;
        TimeSpan _lastTime;

        FXApp()
        {
            //_eff = null; 
            _effectName = null;
            _inited = false;
            _showFPS = false;
            _progress = false;
            _show = false;
            _enableEffect = true;
            _drawVisualization = true;
            _framePeriod = 0.0f;
        }

        ~FXApp()
        {
            NvVFXAPI.NvVFX_DestroyEffect(_eff);
        }

        private string errorStringFromCode(Err code)
        {
            switch (code)
            {
                case Err.errRead:
                    return "There was a problem reading a file";
                case Err.errWrite:
                    return "There was a problem writing a file";
                case Err.errQuit:
                    return "The user chose to quit the application";
                case Err.errFlag:
                    return "There was a problem with the command-line arguments";
                default:
                    if ((int)code <= 0)
                    {
                        return NvCVImageAPI.NvCV_GetErrorStringFromCode((NvCVStatus)code);
                    }
                    else
                    {
                        return "UNKNOWN ERROR";
                    }
            }
        }

        private void drawFrameRate(Mat img)
        {
            //const float timeConstant = 16.0f;
            //std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<float> dur = std::chrono::duration_cast<std::chrono::duration<float>>(now - _lastTime);
            //float t = dur.count();
            //if (0.f < t && t < 100.f)
            //{
            //    if (_framePeriod)
            //        _framePeriod += (t - _framePeriod) * (1.f / timeConstant);  // 1 pole IIR filter
            //    else
            //        _framePeriod = t;
            //    if (_showFPS)
            //    {
            //        char buf[32];
            //        snprintf(buf, sizeof(buf), "%.1f", 1. / _framePeriod);
            //        cv::putText(img, buf, cv::Point(10, img.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
            //    }
            //}
            //else
            //{            // Ludicrous time interval; reset
            //    _framePeriod = 0.f;  // WAKE UP
            //}
            //_lastTime = now;
        }


        private Err processKey(int key, bool FLAG_webcam)
        {
            const int ESC_KEY = 27;
            switch (key)
            {
                case 'Q':
                case 'q':
                case ESC_KEY:
                    return Err.errQuit;
                case 'f':
                case 'F':
                    _showFPS = !_showFPS;
                    break;
                case 'p':
                case 'P':
                case '%':
                    _progress = !_progress;
                    break;
                case 'e':
                case 'E':
                    _enableEffect = !_enableEffect;
                    break;
                case 'd':
                case 'D':
                    if (FLAG_webcam)
                        _drawVisualization = !_drawVisualization;
                    break;
                default:
                    break;
            }

            return Err.errNone;
        }


        Err initCamera(VideoCapture cap, string FLAG_camRes)
        {
            const int camIndex = 0;
            cap.Open(camIndex);
            if (!string.IsNullOrEmpty(FLAG_camRes))
            {
                int camWidth, camHeight, n;

                // parse resolution from string
                var res = FLAG_camRes.Split('x');
                camWidth = Convert.ToInt32(res[0]);
                camHeight = Convert.ToInt32(res[1]);

                //switch (n)
                //{
                //    case 2:
                //        break;  // We have read both width and height
                //    case 1:
                //        camHeight = camWidth;
                //        camWidth = (int)(camHeight * (16.0 / 9.0) + .5);
                //        break;
                //    default:
                //        camHeight = 0;
                //        camWidth = 0;
                //        break;
                //}

                if (camWidth > 0)
                {
                    cap.Set(VideoCaptureProperties.FrameWidth, camWidth);
                }

                if (camHeight > 0)
                {
                    cap.Set(VideoCaptureProperties.FrameHeight, camHeight);
                }

                int actualCamWidth = (int)cap.Get(VideoCaptureProperties.FrameWidth);
                int actualCamHeight = (int)cap.Get(VideoCaptureProperties.FrameHeight);
                if (camWidth != actualCamWidth || camHeight != actualCamHeight)
                {
                    Console.WriteLine("The requested resolution of %d x %d is not available and has been subsituted by %d x %d.\n", camWidth, camHeight, actualCamWidth, actualCamHeight);
                }
            }

            return Err.errNone;
        }


        void drawEffectStatus(Mat img)
        {
            var status = _enableEffect ? "on" : "off";
            string buf = $"Effect: {status}";

            Cv2.PutText(img, buf, new Point(10, img.Rows - 40), HersheyFonts.HersheySimplex, 1, new Scalar(255, 255, 255), 1);
        }

        void CheckResult(NvCVStatus err)
        {
            if (err != NvCVStatus.NVCV_SUCCESS)
            {
                throw new Exception($"NvCVStatus exception. {err.ToString()}");
            }
        }

        Err createEffect(string effectSelector, string modelDir)
        {
            NvCVStatus vfxErr;
            vfxErr = NvVFXAPI.NvVFX_CreateEffect(effectSelector, out _eff);
            CheckResult(vfxErr);

            _effectName = effectSelector;

            if (string.IsNullOrEmpty(modelDir))
            {
                vfxErr = NvVFXAPI.NvVFX_SetString(_eff, NvVFXParameterSelectors.NVVFX_MODEL_DIRECTORY, modelDir);
                CheckResult(vfxErr);
            }
                        
            return appErrFromVfxStatus(vfxErr);
        }

        void destroyEffect()
        {
            NvVFXAPI.NvVFX_DestroyEffect(_eff);
            //_eff = null;
        }

        // Allocate one temp buffer to be used for input and output. Reshaping of the temp buffer in NvCVImage_Transfer() is done automatically,
        // and is very low overhead. We expect the destination to be largest, so we allocate that first to minimize reallocs probablistically.
        // Then we Realloc for the source to get the union of the two.
        // This could alternately be done at runtime by feeding in an empty temp NvCVImage, but there are advantages to allocating all memory at load time.
        NvCVStatus allocTempBuffers()
        {
            NvCVStatus vfxErr;
            vfxErr = NvCVImageAPI.NvCVImage_Alloc(_tmpVFX, _dstVFX.Width, _dstVFX.Height, _dstVFX.PixelFormat, _dstVFX.ComponentType, _dstVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);

            vfxErr = NvCVImageAPI.NvCVImage_Realloc(_tmpVFX, _srcVFX.Width, _srcVFX.Height, _srcVFX.PixelFormat, _srcVFX.ComponentType, _srcVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);
        
            return vfxErr;
        }

        NvCVStatus allocBuffers(uint width, uint height)
        {
            NvCV_Status vfxErr = NVCV_SUCCESS;

            if (_inited)
                return NVCV_SUCCESS;

            if (!_srcImg.data)
            {
                _srcImg.create(height, width, CV_8UC3);                                                                                        // src CPU
                BAIL_IF_NULL(_srcImg.data, vfxErr, NVCV_ERR_MEMORY);
            }

            _dstImg.create(_srcImg.rows, _srcImg.cols, _srcImg.type()); // 
            BAIL_IF_NULL(_dstImg.data, vfxErr, NVCV_ERR_MEMORY); // 
            BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_srcGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1));  // src GPU
            BAIL_IF_ERR(vfxErr = NvCVImage_Alloc(&_dstGpuBuf, _srcImg.cols, _srcImg.rows, NVCV_BGR, NVCV_F32, NVCV_PLANAR, NVCV_GPU, 1)); //dst GPU

            NVWrapperForCVMat(&_srcImg, &_srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(&_dstImg, &_dstVFX);      // _dstVFX is an alias for _dstImg

            //#define ALLOC_TEMP_BUFFERS_AT_RUN_TIME    // Deferring temp buffer allocation is easier
# ifndef ALLOC_TEMP_BUFFERS_AT_RUN_TIME      // Allocating temp buffers at load time avoids run time hiccups
            BAIL_IF_ERR(vfxErr = allocTempBuffers()); // This uses _srcVFX and _dstVFX and allocates one buffer to be a temporary for src and dst
#endif // ALLOC_TEMP_BUFFERS_AT_RUN_TIME

            _inited = true;

        bail:
            return vfxErr;
        }
    }
}
