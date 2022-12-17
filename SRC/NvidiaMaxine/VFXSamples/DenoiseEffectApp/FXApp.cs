using CUDA;
using DenoiseEffectApp;
using NvidiaMaxine.VideoEffects;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DenoiseEffectApp
{
    internal class FXApp
    {
        IntPtr _eff;
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
        public bool _progress;
        bool _enableEffect;
        bool _drawVisualization;
        string _effectName;
        float _framePeriod;
        TimeSpan _lastTime;

        public FXApp()
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

        public string errorStringFromCode(Err code)
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

        public Err createEffect(string effectSelector, string modelDir)
        {
            NvCVStatus vfxErr;
            vfxErr = NvVFXAPI.NvVFX_CreateEffect(effectSelector, out _eff);
            CheckResult(vfxErr);

            _effectName = effectSelector;

            if (!string.IsNullOrEmpty(modelDir))
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
            vfxErr = NvCVImageAPI.NvCVImage_Alloc(ref _tmpVFX, _dstVFX.Width, _dstVFX.Height, _dstVFX.PixelFormat, _dstVFX.ComponentType, _dstVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);

            vfxErr = NvCVImageAPI.NvCVImage_Realloc(_tmpVFX, _srcVFX.Width, _srcVFX.Height, _srcVFX.PixelFormat, _srcVFX.ComponentType, _srcVFX.Planar, NvCVMemSpace.NVCV_GPU, 0);
            CheckResult(vfxErr);

            return vfxErr;
        }

        void NVWrapperForCVMat(Mat cvIm, ref NvCVImage nvcvIm)
        {
            NvCVImagePixelFormat[] nvFormat = new[] { NvCVImagePixelFormat.NVCV_FORMAT_UNKNOWN, NvCVImagePixelFormat.NVCV_Y, NvCVImagePixelFormat.NVCV_YA, NvCVImagePixelFormat.NVCV_BGR, NvCVImagePixelFormat.NVCV_BGRA };
            NvCVImageComponentType[] nvType = new[] { NvCVImageComponentType.NVCV_U8, NvCVImageComponentType.NVCV_TYPE_UNKNOWN, NvCVImageComponentType.NVCV_U16, NvCVImageComponentType.NVCV_S16, NvCVImageComponentType.NVCV_S32,
              NvCVImageComponentType.NVCV_F32, NvCVImageComponentType.NVCV_F64, NvCVImageComponentType.NVCV_TYPE_UNKNOWN };

            nvcvIm.Pixels = cvIm.Data;
            nvcvIm.Width = (uint)cvIm.Cols;
            nvcvIm.Height = (uint)cvIm.Rows;
            nvcvIm.Pitch = (int)cvIm.Step(0);
            nvcvIm.PixelFormat = nvFormat[cvIm.Channels() <= 4 ? cvIm.Channels() : 0];
            nvcvIm.ComponentType = nvType[cvIm.Depth() & 7];
            nvcvIm.BufferBytes = 0;
            //nvcvIm.DeletePtr = null;
            //nvcvIm.DeleteProc = null;
            nvcvIm.PixelBytes = (byte)cvIm.Step(1);
            nvcvIm.ComponentBytes = (byte)cvIm.ElemSize1();
            nvcvIm.NumComponents = (byte)cvIm.Channels();
            nvcvIm.Planar = NvCVLayout.NVCV_CHUNKY;
            nvcvIm.GpuMem = NvCVMemSpace.NVCV_CPU;
            nvcvIm.Reserved1 = 0;
            nvcvIm.Reserved2 = 0;
        }

        NvCVStatus allocBuffers(int width, int height)
        {
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;

            if (_inited)
            {
                return NvCVStatus.NVCV_SUCCESS;
            }

            _srcImg = new Mat();
            if (_srcImg.Data == IntPtr.Zero)
            {
                // src CPU
                _srcImg.Create(height, width, MatType.CV_8UC3);

                if (_srcImg.Data == IntPtr.Zero)
                {
                    return NvCVStatus.NVCV_ERR_MEMORY;
                }
            }

            _dstImg = new Mat();
            _dstImg.Create(_srcImg.Rows, _srcImg.Cols, _srcImg.Type());
            if (_dstImg.Data == IntPtr.Zero)
            {
                return NvCVStatus.NVCV_ERR_MEMORY;
            }

            // src GPU
            _srcGpuBuf = new NvCVImage();
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Cols, (uint)_srcImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //dst GPU
            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_srcImg.Cols, (uint)_srcImg.Rows, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            //#define ALLOC_TEMP_BUFFERS_AT_RUN_TIME    // Deferring temp buffer allocation is easier
            //# ifndef ALLOC_TEMP_BUFFERS_AT_RUN_TIME      // Allocating temp buffers at load time avoids run time hiccups
            CheckResult(allocTempBuffers()); // This uses _srcVFX and _dstVFX and allocates one buffer to be a temporary for src and dst
                                             //#endif // ALLOC_TEMP_BUFFERS_AT_RUN_TIME

            _inited = true;

            return vfxErr;
        }

        public void setShow(bool show) 
        { 
            _show = show; 
        }

        public Err processImage(string inFile, string outFile, float FLAG_strength)
        {
            /*CUstream*/
            IntPtr stream = IntPtr.Zero;
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;

            IntPtr state = IntPtr.Zero;
            IntPtr[] stateArray = new IntPtr[1];

            if (_eff == IntPtr.Zero)
            {
                return Err.errEffect;
            }

            _srcImg = OpenCvSharp.Cv2.ImRead(inFile);
            if (_srcImg.Data == IntPtr.Zero)
            {
                return Err.errRead;
            }

            CheckResult(allocBuffers(_srcImg.Cols, _srcImg.Rows));
            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcGpuBuf, 1.0f / 255.0f, stream, _tmpVFX)); // _srcVFX--> _tmpVFX --> _srcGpuBuf
            CheckResult(NvVFXAPI.NvVFX_SetImage(_eff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE, ref _srcGpuBuf));
            CheckResult(NvVFXAPI.NvVFX_SetImage(_eff, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _dstGpuBuf));
            CheckResult(NvVFXAPI.NvVFX_SetF32(_eff, NvVFXParameterSelectors.NVVFX_STRENGTH, FLAG_strength));

            uint stateSizeInBytes;
            CheckResult(NvVFXAPI.NvVFX_GetU32(_eff, NvVFXParameterSelectors.NVVFX_STATE_SIZE, out stateSizeInBytes));
            CUDA.Runtime.API.cudaMalloc(ref state, stateSizeInBytes);
            CUDA.Runtime.API.cudaMemsetAsync(state, 0, stateSizeInBytes, stream);
            stateArray[0] = state;
            //CheckResult(NvVFXAPI.NvVFX_SetObject(_eff, NvVFXParameterSelectors.NVVFX_STATE, stateArray));
            // BUG!!!

            CheckResult(NvVFXAPI.NvVFX_Load(_eff));
            CheckResult(NvVFXAPI.NvVFX_Run(_eff, 0));
            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstGpuBuf, _dstVFX, 255.0f, stream, _tmpVFX));

            if (!string.IsNullOrEmpty(outFile))
            {
                if (Helpers.IsLossyImageFile(outFile))
                {
                    Console.WriteLine("WARNING: JPEG output file format will reduce image quality\n");
                }

                if (!Cv2.ImWrite(outFile, _dstImg))
                {
                    Console.WriteLine("Error writing: \"%s\"\n", outFile);
                    return Err.errWrite;
                }
            }

            if (_show)
            {
                Cv2.ImShow("Output", _dstImg);
                Cv2.WaitKey(3000);
            }

            if (state != IntPtr.Zero)
            {
                CUDA.Runtime.API.cudaFree(state); // release state memory
            }

            return appErrFromVfxStatus(vfxErr);
        }

        public Err processMovie(Context context)
        {
            int fourcc_h264 = VideoWriter.FourCC('H', '2', '6', '4');
            IntPtr stream = IntPtr.Zero;
            Err appErr = Err.errNone;
            bool ok;
            VideoCapture reader = new VideoCapture();
            VideoWriter writer = new VideoWriter();
            NvCVStatus vfxErr = NvCVStatus.NVCV_SUCCESS;
            uint frameNum;
            VideoInfo info;

            IntPtr state = IntPtr.Zero;            

            if (!context.Webcam && !string.IsNullOrEmpty(context.InFile))
            {
                reader.Open(context.InFile);
            }
            else
            {
                appErr = initCamera(reader, context.CamRes);
                if (appErr != Err.errNone)
                {
                    return appErr;
                }
            }

            if (!reader.IsOpened())
            {
                if (!context.Webcam)
                {
                    Console.WriteLine($"Error: Could not open video: {context.InFile}");
                }
                else
                {
                    Console.WriteLine("Error: Webcam not found\n");
                }

                return Err.errRead;
            }

            Helpers.GetVideoInfo(reader, (!string.IsNullOrEmpty(context.InFile) ? context.InFile : "webcam"), context.Verbose, out info);
            if (!(fourcc_h264 == info.Codec || VideoWriter.FourCC('a', 'v', 'c', '1') == info.Codec)) // avc1 is alias for h264
            {
                Console.WriteLine("Filters only target H264 videos, not %.4s\n", info.Codec);
            }

            CheckResult(allocBuffers(info.Width, info.Height));

            if (!string.IsNullOrEmpty(context.OutFile))
            {
                ok = writer.Open(context.OutFile, VideoWriter.FourCC(context.Codec), info.FrameRate, new Size(_dstVFX.Width, _dstVFX.Height));
                if (!ok)
                {
                    Console.WriteLine("Cannot open \"%s\" for video writing\n", context.OutFile);
                    context.OutFile = null;
                    if (!_show)
                    {
                        return Err.errWrite;
                    }
                }
            }

            
            CheckResult(NvVFXAPI.NvVFX_SetImage(_eff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE, ref _srcGpuBuf));
            CheckResult(NvVFXAPI.NvVFX_SetImage(_eff, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _dstGpuBuf));

            CheckResult(NvVFXAPI.NvVFX_SetF32(_eff, NvVFXParameterSelectors.NVVFX_STRENGTH, context.Strength));

            uint stateSizeInBytes;
            CheckResult(NvVFXAPI.NvVFX_GetU32(_eff, NvVFXParameterSelectors.NVVFX_STATE_SIZE, out stateSizeInBytes));
            CUDA.Runtime.API.cudaMalloc(ref state, stateSizeInBytes);
            CUDA.Runtime.API.cudaMemsetAsync(state, 0, stateSizeInBytes, stream);

            IntPtr[] stateArray = new IntPtr[1];
            stateArray[0] = state;
            IntPtr buffer = Marshal.AllocCoTaskMem(Marshal.SizeOf(typeof(IntPtr)) * stateArray.Length);
            Marshal.Copy(stateArray, 0, buffer, stateArray.Length);
                                              
            CheckResult(NvVFXAPI.NvVFX_SetObject(_eff, NvVFXParameterSelectors.NVVFX_STATE, buffer));
            CheckResult(NvVFXAPI.NvVFX_Load(_eff));

            for (frameNum = 0; reader.Read(_srcImg); frameNum++)
            {
                if (_enableEffect)
                {
                    CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcGpuBuf, 1.0f / 255.0f, stream, _tmpVFX));
                    CheckResult(NvVFXAPI.NvVFX_Run(_eff, 0));
                    CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstGpuBuf, _dstVFX, 255.0f, stream, _tmpVFX));
                }
                else
                {
                    CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _dstVFX, 1.0f, stream, _tmpVFX));
                    CUDA.Runtime.API.cudaMemsetAsync(state, 0, stateSizeInBytes, stream);// reset state by setting to 0
                }

                if (!string.IsNullOrEmpty(context.OutFile))
                {
                    writer.Write(_dstImg);
                }

                if (_show)
                {
                    if (_drawVisualization) drawEffectStatus(_dstImg);
                    drawFrameRate(_dstImg);
                    Cv2.ImShow("Output", _dstImg);
                    int key = Cv2.WaitKey(1);
                    if (key > 0)
                    {
                        appErr = processKey(key, context.Webcam);
                        if (Err.errQuit == appErr)
                            break;
                    }
                }
                if (_progress)
                {
                    Console.WriteLine("\b\b\b\b%3.0f%%", 100.0f * frameNum / info.FrameCount);
                }
            }

            reader.Release();
            if (!string.IsNullOrEmpty(context.OutFile))
            {
                writer.Release();
            }

            if (state != IntPtr.Zero)
            {
                CUDA.Runtime.API.cudaFree(state); // release state memory
            }

            return appErrFromVfxStatus(vfxErr);
        }

        private Err appErrFromVfxStatus(NvCVStatus status)
        {
            return (Err)status;
        }
    }
}
