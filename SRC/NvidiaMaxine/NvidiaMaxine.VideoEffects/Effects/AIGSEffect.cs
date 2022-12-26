// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : Roman Miniailov
// Created          : 12-26-2022
//
// Last Modified By : Roman Miniailov
// Last Modified On : 12-26-2022
// ***********************************************************************
// <copyright file="AIGSEffect.cs" company="">
//     Copyright (c) 2006-2022
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;

#if OPENCV
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
#endif

namespace NvidiaMaxine.VideoEffects.Effects
{
    /// <summary>
    /// AI Green Screen effect.
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />.
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    public class AIGSEffect : BaseEffect
    {
        private readonly uint _maxInputWidth = 3840;

        private readonly uint _maxInputHeight = 2160;

        private readonly uint _maxNumberStreams = 1;

        private bool _cudaGraph;

        private readonly List<IntPtr> _stateArray = new List<IntPtr>();

        private IntPtr _bgblurEff;

        private IntPtr _batchOfStates;

        private uint _modelBatch;

#if OPENCV
        private Mat _bgImg;

        private Mat _resizedCroppedBgImg;
        
        private Mat _result;
#else
        private VideoFrame _bgImg;

        private VideoFrame _resizedCroppedBgImg;

        private VideoFrame _result;
#endif

        private NvCVImage _srcNvVFXImage;

        private NvCVImage _dstNvVFXImage;

        private NvCVImage _blurNvVFXImage;

        private long _count;

        private long _total;

        public uint Mode { get; set; }

        /// <summary>
        /// Gets or sets the blur strength.
        /// </summary>
        public float BlurStrength { get; set; } = 0.5f;

        /// <summary>
        /// Gets or sets the effect mode.
        /// </summary>
        public AIGSEffectMode EffectMode { get; set; } = AIGSEffectMode.Background;

        /// <summary>
        /// Gets or sets the background image.
        /// </summary>
        public string BackgroundImage { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AIGSEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
        /// <param name="effectMode">The effect mode.</param>
#if OPENCV
        public AIGSEffect(string modelsDir, Mat sourceImage, AIGSEffectMode effectMode)
            : base(NvVFXFilterSelectors.NVVFX_FX_GREEN_SCREEN, modelsDir, sourceImage)
#else
        public AIGSEffect(string modelsDir, VideoFrame sourceImage, AIGSEffectMode effectMode)
            : base(NvVFXFilterSelectors.NVVFX_FX_GREEN_SCREEN, modelsDir, sourceImage)
#endif
        {
            EffectMode = effectMode;

            // Choose one mode -> set() -> Load() -> Run()
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MODE, Mode));
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_CUDA_GRAPH, _cudaGraph ? 1u : 0u));
            CheckResult(NvVFXAPI.NvVFX_CudaStreamCreate(out _stream));
            CheckResult(NvVFXAPI.NvVFX_SetCudaStream(_handle, NvVFXParameterSelectors.NVVFX_CUDA_STREAM, _stream));

            // Set maximum width, height and number of streams and then call Load() again
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MAX_INPUT_WIDTH, _maxInputWidth));
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MAX_INPUT_HEIGHT, _maxInputHeight));
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MAX_NUMBER_STREAMS, _maxNumberStreams));

            CheckResult(NvVFXAPI.NvVFX_Load(_handle));

            for (uint i = 0; i < _maxNumberStreams; i++)
            {
                IntPtr state;
                CheckResult(NvVFXAPI.NvVFX_AllocateState(_handle, out state));

                _stateArray.Add(state);
            }

            CreateBackgroundBlurEffect();
        }

        private void CreateBackgroundBlurEffect()
        {
            // Create Background blur effect

            CheckResult(NvVFXAPI.NvVFX_CreateEffect(NvVFXFilterSelectors.NVVFX_FX_BGBLUR, out _bgblurEff));
            //CheckResult(NvVFXAPI.NvVFX_GetString(_bgblurEff, NvVFXParameterSelectors.NVVFX_INFO, cstr));
            CheckResult(NvVFXAPI.NvVFX_SetCudaStream(_bgblurEff, NvVFXParameterSelectors.NVVFX_CUDA_STREAM, _stream));
        }

        /// <summary>
        /// Applies the effect.
        /// </summary>
        protected override void ApplyEffect()
        {
            CheckResult(NvVFXAPI.NvVFX_SetU32(_handle, NvVFXParameterSelectors.NVVFX_MODE, (uint)Mode));
        }

        /// <summary>
        /// Destroys the effect.
        /// </summary>
        protected override void DestroyEffect()
        {
            // If DeallocateState fails, all memory allocated in the SDK returns to the heap when the effect handle is destroyed.
            for (int i = 0; i < _stateArray.Count; i++)
            {
                NvVFXAPI.NvVFX_DeallocateState(_handle, _stateArray[i]);
            }

            _stateArray.Clear();

            if (_batchOfStates != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_batchOfStates);
                _batchOfStates = IntPtr.Zero;
            }

            base.DestroyEffect();

            NvVFXAPI.NvVFX_DestroyEffect(_bgblurEff);
            _bgblurEff = IntPtr.Zero;

            if (_stream != IntPtr.Zero)
            {
                NvVFXAPI.NvVFX_CudaStreamDestroy(_stream);
            }
        }

        private bool LoadBackgroundImage(int width, int height)
        {
            if (string.IsNullOrEmpty(BackgroundImage))
            {
                return false;
            }

            if (!File.Exists(BackgroundImage))
            {
                throw new FileNotFoundException("Background image not found", BackgroundImage);
            }

#if OPENCV
            _bgImg = Cv2.ImRead(BackgroundImage);
#else
            _bgImg = VideoFrame.LoadFromFile(BackgroundImage);
#endif
            
            if (_bgImg.Data == IntPtr.Zero)
            {
                throw new Exception("Background image not loaded");
            }
            else
            {
#if OPENCV         
                // Find the scale to resize background such that image can fit into background
                float scale = (float)height / (float)_bgImg.Height;
                if ((scale * _bgImg.Width) < (float)width)
                {
                    scale = (float)width / (float)_bgImg.Width;
                }
                var resizedBg = new Mat();
                Cv2.Resize(_bgImg, resizedBg, new Size(), scale, scale, InterpolationFlags.Area);
#else
                var resizedBg = _bgImg.ResizeImage24(width, height);
#endif

                // Always crop from top left of background.                

#if OPENCV
                var rect = new Rect(0, 0, width, height);
                _resizedCroppedBgImg = new Mat(resizedBg, rect);
#else
                var rect = new Rectangle(0, 0, width, height);
                _resizedCroppedBgImg = ImageHelper.Crop(resizedBg, rect);
#endif
            }

            return true;
        }

        /// <summary>
        /// Allocs the buffers.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>NvCVStatus.</returns>
        protected override NvCVStatus AllocBuffers(int width, int height)
        {
            LoadBackgroundImage(width, height);

            _modelBatch = 1;

            // Allocate space for batchOfStates to hold state variable addresses
            // Assume that MODEL_BATCH Size is enough for this scenario
            CheckResult(NvVFXAPI.NvVFX_GetU32(_handle, NvVFXParameterSelectors.NVVFX_MODEL_BATCH, out _modelBatch));
            _batchOfStates = Marshal.AllocHGlobal((int)_modelBatch * IntPtr.Size);
            if (_batchOfStates == IntPtr.Zero)
            {
                CheckResult(NvCVStatus.NVCV_ERR_MEMORY);
            }

            // allocate src for GPU
            if (_srcNvVFXImage.Pixels == IntPtr.Zero)
            {
                CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcNvVFXImage, (uint)width, (uint)height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8, NvCVLayout.NVCV_CHUNKY, NvCVMemSpace.NVCV_GPU, 1));
            }

            // allocate dst for GPU
            if (_dstNvVFXImage.Pixels == IntPtr.Zero)
            {
                CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstNvVFXImage, (uint)width, (uint)height, NvCVImagePixelFormat.NVCV_A, NvCVImageComponentType.NVCV_U8, NvCVLayout.NVCV_CHUNKY, NvCVMemSpace.NVCV_GPU, 1));
            }

            // allocate blur for GPU
            if (_blurNvVFXImage.Pixels == IntPtr.Zero)
            {
                CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _blurNvVFXImage, (uint)width, (uint)height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8, NvCVLayout.NVCV_CHUNKY, NvCVMemSpace.NVCV_GPU, 1));
            }

            //            if (_srcImg == null || _srcImg.Data == IntPtr.Zero)
            //            {
            //                // src CPU
            //#if OPENCV
            //                _srcImg = new Mat();
            //                _srcImg.Create(height, width, MatType.CV_8UC3);
            //#else
            //                _srcImg = new VideoFrame(width, height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);
            //#endif

            //                if (_srcImg.Data == IntPtr.Zero)
            //                {
            //                    return NvCVStatus.NVCV_ERR_MEMORY;
            //                }
            //            }
            //            else
            //            {
            //#if !OPENCV
            //                if (_srcImg.PixelFormat != NvCVImagePixelFormat.NVCV_BGR || _srcImg.ComponentType != NvCVImageComponentType.NVCV_U8)
            //                {
            //                    return NvCVStatus.NVCV_ERR_PARAMETER;
            //                }
            //#endif
            //            }

            //#if OPENCV
            //            _dstImg = new Mat();
            //            _dstImg.Create(_srcImg.Height, _srcImg.Width, _srcImg.Type());
            //#else
            //            _dstImg = new VideoFrame(_srcImg.Width, _srcImg.Height, _srcImg.PixelFormat, _srcImg.ComponentType);
            //#endif

            //            if (_dstImg.Data == IntPtr.Zero)
            //            {
            //                return NvCVStatus.NVCV_ERR_MEMORY;
            //            }

            //            // src GPU
            //            _srcGpuBuf = new NvCVImage();
            //            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _srcGpuBuf, (uint)_srcImg.Width, (uint)_srcImg.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //            //dst GPU
            //            _dstGpuBuf = new NvCVImage();
            //            CheckResult(NvCVImageAPI.NvCVImage_Alloc(ref _dstGpuBuf, (uint)_dstImg.Width, (uint)_dstImg.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_F32, NvCVLayout.NVCV_PLANAR, NvCVMemSpace.NVCV_GPU, 1));

            //            NVWrapperForCVMat(_srcImg, ref _srcVFX);      // _srcVFX is an alias for _srcImg
            //            NVWrapperForCVMat(_dstImg, ref _dstVFX);      // _dstVFX is an alias for _dstImg

            //            CheckResult(AllocTempBuffers());

            return NvCVStatus.NVCV_SUCCESS;
        }

        /// <summary>
        /// Overlays the specified image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="mask">The mask.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="result">The result.</param>
#if OPENCV
        private static void Overlay(Mat image, Mat mask, float alpha, out Mat result)
#else
        private static void Overlay(VideoFrame image, VideoFrame mask, float alpha, out VideoFrame result)
#endif
        {
#if OPENCV
            Mat maskClr = new Mat();
            Cv2.CvtColor(mask, maskClr, ColorConversionCodes.GRAY2BGR);
            result = image * (1.0f - alpha) + maskClr * alpha;
            maskClr.Dispose();
#else
            throw new NotImplementedException("Overlay not implemented yet.");
#endif
        }

        /// <summary>
        /// Initializes.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>NvCVStatus.</returns>
        public override NvCVStatus Init(int width, int height)
        {
            _state = IntPtr.Zero;
            _stream = IntPtr.Zero;

            CheckResult(AllocBuffers(width, height));

#if OPENCV
            _dstImg = new Mat(_srcImg.Size(), MatType.CV_8UC1);
            _result = new Mat();
            _result.Create(_srcImg.Rows, _srcImg.Cols, MatType.CV_8UC3);  
#else
            _dstImg = new VideoFrame(_srcImg.Width, _srcImg.Height, NvCVImagePixelFormat.NVCV_Y, NvCVImageComponentType.NVCV_U8);
            _result = new VideoFrame(_srcImg.Width, _srcImg.Height, NvCVImagePixelFormat.NVCV_BGR, NvCVImageComponentType.NVCV_U8);
#endif
            
            CheckNull(_result.Data, NvCVStatus.NVCV_ERR_MEMORY);

            _count = 0;
            _total = 0;

            return NvCVStatus.NVCV_SUCCESS;
        }

        /// <summary>
        /// Processes.
        /// </summary>
        /// <returns>Returns processed frame.</returns>
#if OPENCV
        public override Mat Process()
#else
        public override VideoFrame Process()
#endif
        {
            //CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcGpuBuf, 1.0f / 255.0f, _stream, _tmpVFX));
            //CheckResult(NvVFXAPI.NvVFX_Run(_handle, 0));
            //CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstGpuBuf, _dstVFX, 255.0f, _stream, _tmpVFX));

            //return _dstImg;

#if OPENCV
            _dstImg.SetTo(new Scalar(0));
#else
            _dstImg.Clear();
#endif
            
            NVWrapperForCVMat(_srcImg, ref _srcVFX);
            NVWrapperForCVMat(_dstImg, ref _dstVFX);

            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE, ref _srcNvVFXImage));
            CheckResult(NvVFXAPI.NvVFX_SetImage(_handle, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _dstNvVFXImage));
            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcNvVFXImage, 1.0f, _stream, IntPtr.Zero));

            // Assign states from stateArray in batchOfStates
            // There is only one stream in this app

            IntPtr[] stateArray = new IntPtr[_stateArray.Count];
            for (int i = 0; i < _stateArray.Count; i++)
            {
                stateArray[i] = _stateArray[i];
            }

            _batchOfStates = Marshal.AllocCoTaskMem(Marshal.SizeOf(typeof(IntPtr)) * stateArray.Length);
            Marshal.WriteIntPtr(_batchOfStates, _stateArray[0]);

            CheckResult(NvVFXAPI.NvVFX_SetStateObjectHandleArray(_handle, NvVFXParameterSelectors.NVVFX_STATE, _batchOfStates));

            Stopwatch sw = new Stopwatch();
            sw.Start();
            CheckResult(NvVFXAPI.NvVFX_Run(_handle, 0));
            sw.Stop();
            var ms = sw.ElapsedMilliseconds;
            _count += 1;
            if (_count > 0)
            {
                // skipping first frame
                _total += ms;
            }

            CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstNvVFXImage, _dstVFX, 1.0f, _stream, IntPtr.Zero));

            NvCVImage matVFX = new NvCVImage();

#if OPENCV
            _result.SetTo(Scalar.All(0));  
#else
            _result.Clear();
#endif
            
            switch (EffectMode)
            {
                case AIGSEffectMode.None:
                    _srcImg.CopyTo(_result);
                    break;

                case AIGSEffectMode.Background:
                    {
                        if (string.IsNullOrEmpty(BackgroundImage))
                        {
#if OPENCV
                            _resizedCroppedBgImg = new Mat(_srcImg.Rows, _srcImg.Cols, MatType.CV_8UC3, new Scalar(118, 185, 0));
                            var startX = _resizedCroppedBgImg.Cols / 20;
                            var offsetY = _resizedCroppedBgImg.Rows / 20;
                            string text = "No Background Image!";
                            for (var startY = offsetY; startY < _resizedCroppedBgImg.Height; startY += offsetY)
                            {
                                Cv2.PutText(_resizedCroppedBgImg, text, new Point(startX, startY),
                                    HersheyFonts.HersheyDuplex, 1.0, Scalar.FromRgb(0, 0, 0), 1);
                            }
#endif
                        }

                        NvCVImage bgVFX = new NvCVImage();
                        NVWrapperForCVMat(_resizedCroppedBgImg, ref bgVFX);
                        NVWrapperForCVMat(_result, ref matVFX);
                        NvCVImageAPI.NvCVImage_Composite(_srcVFX, bgVFX, _dstVFX, out matVFX, _stream);
                    }

                    break;
                    
#if OPENCV
                case AIGSEffectMode.Light:
                    //if (inFile)
                    //{
                    Overlay(_srcImg, _dstImg, 0.5f, out var result);
                    result.CopyTo(_result);
                    result.Dispose();
                    //}
                    //else
                    //{  
                    //    // If the webcam was cropped, also crop the compositing
                    //    Rect rect = new Rect(0, (int)((_srcImg.Rows - _srcVFX.Height) / 2), (int)_srcVFX.Width, (int)_srcVFX.Height);
                    //    Mat subResult = new Mat(result, rect);
                    //    overlay(new Mat(_srcImg, rect), new Mat(_dstImg, rect), 0.5, subResult);
                    //}

                    break;
#endif

                case AIGSEffectMode.Green:
                    {
                        int bgColor = System.Drawing.Color.Green.ToArgb();
                        NVWrapperForCVMat(_result, ref matVFX);
                        NvCVImageAPI.NvCVImage_CompositeOverConstant(_srcVFX, _dstVFX, ref bgColor, ref matVFX, _stream);
                    }

                    break;

                case AIGSEffectMode.White:
                    {
                        int bgColor = System.Drawing.Color.White.ToArgb();
                        NVWrapperForCVMat(_result, ref matVFX);
                        NvCVImageAPI.NvCVImage_CompositeOverConstant(_srcVFX, _dstVFX, ref bgColor, ref matVFX, _stream);
                    }

                    break;

                case AIGSEffectMode.Matte:
                    //Cv2.CvtColor(_dstImg, _result, ColorConversionCodes.GRAY2BGR);
                    var res = ImageHelper.ConvertGrayscaleToRGB(_dstImg);
                    res.CopyTo(_result);
                    res.Dispose();
                    
                    break;

                case AIGSEffectMode.Blur:

                    CheckResult(NvVFXAPI.NvVFX_SetF32(_bgblurEff, NvVFXParameterSelectors.NVVFX_STRENGTH, BlurStrength));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE_0, ref _srcNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE_1, ref _dstNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _blurNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_Load(_bgblurEff));
                    CheckResult(NvVFXAPI.NvVFX_Run(_bgblurEff, 0));

                    NVWrapperForCVMat(_result, ref matVFX);
                    CheckResult(NvCVImageAPI.NvCVImage_Transfer(_blurNvVFXImage, matVFX, 1.0f, _stream, IntPtr.Zero));

                    break;
            }

            return _result;

            //if (outFile)
            //{
            //    //#define WRITE_COMPOSITE
            //    writer.write(result);
            //    //#else   // WRITE_MATTE
            //    //                writer.write(_dstImg);

            //}

            //if (_show)
            //{
            //    drawFrameRate(result);
            //    cv::imshow("Output", result);
            //    int key = cv::waitKey(1);
            //    if (key > 0)
            //    {
            //        appErr = processKey(key);
            //        if (errQuit == appErr) break;
            //    }
            //}
            //if (_progress)
            //    if (info.frameCount == 0)  // no progress for a webcam
            //        fprintf(stderr, "\b\b\b\b???%%");
            //    else
            //        fprintf(stderr, "\b\b\b\b%3.0f%%", 100.f * frameNum / info.frameCount);
        }
    }
}
