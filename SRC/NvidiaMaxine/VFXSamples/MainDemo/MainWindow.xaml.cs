using NvidiaMaxine.VideoEffects;
using NvidiaMaxine.VideoEffects.Effects;
using NvidiaMaxine.VideoEffects.Outputs;
using NvidiaMaxine.VideoEffects.Sources;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace MainDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private IBaseSource _source;

        private FileOutput _output;

        private DenoiseEffect _denoiseEffect;

        private ArtifactReductionEffect _artifactReductionEffect;

        private SuperResolutionEffect _superResolutionEffect;

        private UpscaleEffect _upscaleEffect;

        private BaseEffect _currentEffect;

        private ulong _frameID;

        private ulong _totalFrames;

        private int _lastProgress;
        
        private WriteableBitmap _previewBitmap;

        private OpenCvSharp.Mat _previewMat;

        private string MODELS_DIR = @"c:\Projects\_Projects\NvidiaMaxineNet\SDK\bin\models\";

        public MainWindow()
        {
            InitializeComponent();
        }

        private void btStart_Click(object sender, RoutedEventArgs e)
        {
            _frameID = 0;
            _lastProgress = 0;

            // add source
            VideoInfo info = new VideoInfo();
            if (rbFile.IsChecked == true)
            {
                if (!File.Exists(edSourceFilename.Text))
                {
                    MessageBox.Show("Unable to find source file.");
                    return;
                }

                _source = new FileVideoSource();
                (_source as FileVideoSource).Open(edSourceFilename.Text);
                _source.FrameReady += Source_FrameReady;
                _source.Complete += Source_Complete;

                _source.GetVideoInfo(out info);
                _totalFrames = (ulong)info.FrameCount;
            }

            // add output
            _output = new FileOutput();
            _output.Init(edOutputFilename.Text, info.Resolution, info.FrameRate);

            // add effect
            if (rbEffDenoise.IsChecked == true)
            {
                _denoiseEffect = new DenoiseEffect(MODELS_DIR, _source.GetBaseFrame());
                _denoiseEffect.Init(info.Width, info.Height);
                _currentEffect = _denoiseEffect;
            }
            else if (rbEffArtReduction.IsChecked == true)
            {
                _artifactReductionEffect = new ArtifactReductionEffect(MODELS_DIR, _source.GetBaseFrame());
                _artifactReductionEffect.Init(info.Width, info.Height);
                _currentEffect = _artifactReductionEffect;
            }
            else if (rbEffSuperRes.IsChecked == true)
            {
                _superResolutionEffect = new SuperResolutionEffect(MODELS_DIR, _source.GetBaseFrame());
                _superResolutionEffect.Init(info.Width, info.Height);
                _currentEffect = _superResolutionEffect;
            }
            else if (rbEffUpscale.IsChecked == true)
            {
                _upscaleEffect = new UpscaleEffect(MODELS_DIR, _source.GetBaseFrame());
                _upscaleEffect.Init(info.Width, info.Height);
                _currentEffect = _upscaleEffect;
            }                       

            // start
            _source.Start();
        }

        private void Source_Complete(object sender, EventArgs e)
        {
            StopAll();

            MessageBox.Show("Done.");
        }

        private void Source_FrameReady(object sender, VideoFrameEventArgs e)
        {
            //OpenCvSharp.Cv2.ImWrite("c:\\vf\\x\\orig.jpg", e.Frame);

            var processedFrame = _currentEffect.Process();
            _output.WriteFrame(processedFrame);

            //OpenCvSharp.Cv2.ImWrite("c:\\vf\\x\\proc.jpg", processedFrame);

            var progress = (int)((_frameID * 100) / _totalFrames);
            if (progress != _lastProgress)
            {
                _lastProgress = progress;
                UpdateProgress(progress);
            }

            Dispatcher.Invoke(() =>
            {
                if (cbPreview.IsChecked == true)
                {
                    RenderFrame(processedFrame);
                }                
            });

            _frameID++;
        }

        private void UpdateProgress(int progress)
        {
            Debug.WriteLine($"Progress: {_lastProgress}%");

            Dispatcher.Invoke(() =>
            {
                pbProgress.Value = progress;
            });
        }

        private void StopAll()
        {
            _source?.Stop();
            _source?.Dispose();
            _source = null;

            _output?.Finish();
            _output?.Dispose();
            _output = null;

            _denoiseEffect?.Dispose();
            _denoiseEffect = null;

            _artifactReductionEffect?.Dispose();
            _artifactReductionEffect = null;

            _upscaleEffect?.Dispose();
            _upscaleEffect = null;

            _superResolutionEffect?.Dispose();
            _superResolutionEffect = null;

            _currentEffect = null;

            Dispatcher.Invoke(() =>
            {
                pbProgress.Value = 0;
            });
        }

        private void RenderFrame(OpenCvSharp.Mat frame)
        {
            if (_previewBitmap == null || _previewBitmap.PixelWidth != frame.Width || _previewBitmap.PixelHeight != frame.Height || pnScreen.Source == null)
            {
                var dpi = VisualTreeHelper.GetDpi(pnScreen);
                _previewBitmap = new WriteableBitmap(frame.Width, frame.Height, dpi.PixelsPerInchX, dpi.PixelsPerInchY, PixelFormats.Bgr24, null);

                pnScreen.BeginInit();
                pnScreen.Source = this._previewBitmap;
                pnScreen.EndInit();

                _previewMat = new OpenCvSharp.Mat(frame.Height, frame.Width, OpenCvSharp.MatType.CV_8UC3);
            }

            frame.ConvertTo(_previewMat, OpenCvSharp.MatType.CV_8UC3);

            pnScreen.BeginInit();
            int lineStep = (((frame.Width * 24) + 31) / 32) * 4;
            _previewBitmap.WritePixels(new Int32Rect(0, 0, frame.Width, frame.Height), frame.Data, lineStep * frame.Height, lineStep);
            pnScreen.EndInit();
        }

        private void btStop_Click(object sender, RoutedEventArgs e)
        {
            StopAll();
        }
    }
}
