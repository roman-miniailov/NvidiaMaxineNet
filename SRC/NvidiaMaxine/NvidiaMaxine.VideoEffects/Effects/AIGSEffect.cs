// ***********************************************************************
// Assembly         : NvidiaMaxine.VideoEffects
// Author           : roman
// Created          : 12-21-2022
//
// Last Modified By : roman
// Last Modified On : 12-21-2022
// ***********************************************************************
// <copyright file="AIGSEffect.cs" company="NvidiaMaxine.VideoEffects">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************

using NvidiaMaxine.VideoEffects.API;

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
    /// Implements the <see cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    /// </summary>
    /// <seealso cref="NvidiaMaxine.VideoEffects.Effects.BaseEffect" />
    public class AIGSEffect : BaseEffect
    {
        private uint _maxInputWidth = 3840;

        private uint _maxInputHeight = 2160;

        private uint _maxNumberStreams = 1;

        private bool _cudaGraph;

        private List<IntPtr> _stateArray = new List<IntPtr>();

        private IntPtr _bgblurEff;

        private IntPtr _batchOfStates;

        private uint _modelBatch;

        private Mat _bgImg;

        private Mat _resizedCroppedBgImg;

        private NvCVImage _srcNvVFXImage;

        private NvCVImage _dstNvVFXImage;

        private NvCVImage _blurNvVFXImage;

        private float _blurStrength = 0.5f;

        private long _count;

        private long _total;




        public uint Mode { get; set; }

        public AIGSEffectMode EffectMode { get; set; } = AIGSEffectMode.Background;

        public string BackgroundImage { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AIGSEffect"/> class.
        /// </summary>
        /// <param name="modelsDir">The models dir.</param>
        /// <param name="sourceImage">The source image.</param>
#if OPENCV
        public AIGSEffect(string modelsDir, Mat sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_GREEN_SCREEN, modelsDir, sourceImage)
#else
        public AIGSEffect(string modelsDir, VideoFrame sourceImage) : base(NvVFXFilterSelectors.NVVFX_FX_GREEN_SCREEN, modelsDir, sourceImage)
#endif
        {

            //const char* cstr;  // TODO: This is not necessary
            //vfxErr = NvVFX_GetString(_eff, NVVFX_INFO, &cstr);
            //if (vfxErr != NVCV_SUCCESS)
            //{
            //    std::cerr << "AIGS modes not found \n" << std::endl;
            //    return vfxErr;
            //}

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
            // ------------------ create Background blur effect ------------------ //
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

            _bgImg = Cv2.ImRead(BackgroundImage);
            if (_bgImg.Data == IntPtr.Zero)
            {
                throw new Exception("Background image not loaded");
            }
            else
            {
                // Find the scale to resize background such that image can fit into background
                float scale = (float)height / (float)_bgImg.Height;
                if ((scale * _bgImg.Width) < (float)width)
                {
                    scale = (float)width / (float)_bgImg.Width;
                }

                Mat resizedBg = new Mat();
                Cv2.Resize(_bgImg, resizedBg, new Size(), scale, scale, InterpolationFlags.Area);

                // Always crop from top left of background.
                Rect rect = new Rect(0, 0, width, height);

                _resizedCroppedBgImg = new Mat(resizedBg, rect);
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
        private static void Overlay(Mat image, Mat mask, float alpha, Mat result)
        {
            Mat maskClr = new Mat();
            Cv2.CvtColor(mask, maskClr, ColorConversionCodes.GRAY2BGR);
            result = image * (1.0f - alpha) + maskClr * alpha;
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

            _dstImg = new Mat(_srcImg.Size(), MatType.CV_8UC1);

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
            Mat result = new Mat();
            //CheckResult(NvCVImageAPI.NvCVImage_Transfer(_srcVFX, _srcGpuBuf, 1.0f / 255.0f, _stream, _tmpVFX));
            //CheckResult(NvVFXAPI.NvVFX_Run(_handle, 0));
            //CheckResult(NvCVImageAPI.NvCVImage_Transfer(_dstGpuBuf, _dstVFX, 255.0f, _stream, _tmpVFX));

            //return _dstImg;

            _dstImg.SetTo(new Scalar(0));

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

            result.Create(_srcImg.Rows, _srcImg.Cols, MatType.CV_8UC3);  // Make sure the result is allocated. TODO: allocate outsifde of the loop?
            CheckNull(result.Data, NvCVStatus.NVCV_ERR_MEMORY);
            result.SetTo(Scalar.All(0));  // TODO: This may no longer be necessary since we no longer coerce to 16:9
            switch (EffectMode)
            {
                case AIGSEffectMode.None:
                    _srcImg.CopyTo(result);
                    break;

                case AIGSEffectMode.Background:
                    {
                        if (string.IsNullOrEmpty(BackgroundImage))
                        {
                            _resizedCroppedBgImg = new Mat(_srcImg.Rows, _srcImg.Cols, MatType.CV_8UC3, new Scalar(118, 185, 0));
                            var startX = _resizedCroppedBgImg.Cols / 20;
                            var offsetY = _resizedCroppedBgImg.Rows / 20;
                            string text = "No Background Image!";
                            for (var startY = offsetY; startY < _resizedCroppedBgImg.Rows; startY += offsetY)
                            {
                                Cv2.PutText(_resizedCroppedBgImg, text, new Point(startX, startY),
                                    HersheyFonts.HersheyDuplex, 1.0, Scalar.FromRgb(0, 0, 0), 1);
                            }
                        }

                        NvCVImage bgVFX = new NvCVImage();
                        NVWrapperForCVMat(_resizedCroppedBgImg, ref bgVFX);
                        NVWrapperForCVMat(result, ref matVFX);
                        NvCVImageAPI.NvCVImage_Composite(_srcVFX, bgVFX, _dstVFX, out matVFX, _stream);
                    }

                    break;

                case AIGSEffectMode.Light:
                    //if (inFile)
                    //{
                        Overlay(_srcImg, _dstImg, 0.5f, result);
                    //}
                    //else
                    //{  
                    //    // If the webcam was cropped, also crop the compositing
                    //    Rect rect = new Rect(0, (int)((_srcImg.Rows - _srcVFX.Height) / 2), (int)_srcVFX.Width, (int)_srcVFX.Height);
                    //    Mat subResult = new Mat(result, rect);
                    //    overlay(new Mat(_srcImg, rect), new Mat(_dstImg, rect), 0.5, subResult);
                    //}

                    break;

                case AIGSEffectMode.Green:
                    {
                        int bgColor = System.Drawing.Color.Green.ToArgb();
                        NVWrapperForCVMat(result, ref matVFX);
                        NvCVImageAPI.NvCVImage_CompositeOverConstant(_srcVFX, _dstVFX, ref bgColor, ref matVFX, _stream);
                    }

                    break;

                case AIGSEffectMode.White:
                    {
                        int bgColor = System.Drawing.Color.White.ToArgb();
                        NVWrapperForCVMat(result, ref matVFX);
                        NvCVImageAPI.NvCVImage_CompositeOverConstant(_srcVFX, _dstVFX, ref bgColor, ref matVFX, _stream);
                    }

                    break;

                case AIGSEffectMode.Matte:
                    Cv2.CvtColor(_dstImg, result, ColorConversionCodes.GRAY2BGR);
                    break;

                case AIGSEffectMode.Blur:

                    CheckResult(NvVFXAPI.NvVFX_SetF32(_bgblurEff, NvVFXParameterSelectors.NVVFX_STRENGTH, _blurStrength));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE_0, ref _srcNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_INPUT_IMAGE_1, ref _dstNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_SetImage(_bgblurEff, NvVFXParameterSelectors.NVVFX_OUTPUT_IMAGE, ref _blurNvVFXImage));
                    CheckResult(NvVFXAPI.NvVFX_Load(_bgblurEff));
                    CheckResult(NvVFXAPI.NvVFX_Run(_bgblurEff, 0));

                    NVWrapperForCVMat(result, ref matVFX);
                    CheckResult(NvCVImageAPI.NvCVImage_Transfer(_blurNvVFXImage, matVFX, 1.0f, _stream, IntPtr.Zero));

                    break;
            }

            return result;

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
