using NvidiaMaxine.VideoEffects;
using NvidiaMaxine.VideoEffects.Effects;
using NvidiaMaxine.VideoEffects.Outputs;
using NvidiaMaxine.VideoEffects.Sources;

using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace MainDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private IBaseSource _source;

        private FileOutput _output;

        private BaseEffect _videoEffect;

        private bool _stopFlag = false;

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
            _stopFlag = false;

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
            else
            {
                _source = new CameraVideoSource();
                (_source as CameraVideoSource).Open(cbCamera.Text);
                _source.FrameReady += Source_FrameReady;

                _source.GetVideoInfo(out info);
                _totalFrames = 0;
            }

            // add output
            _output = new FileOutput();
            _output.Init(edOutputFilename.Text, info.Resolution, info.FrameRate);

            // add effect
            switch (cbEffect.SelectedIndex)
            {
                case 0:
                    {
                        _videoEffect = new DenoiseEffect(MODELS_DIR, _source.GetBaseFrame(), (float)(slDenoiseStrength.Value / 10.0));
                        _videoEffect.Init(info.Width, info.Height);
                    }
                    
                    break;
                case 1:
                    {
                        _videoEffect = new ArtifactReductionEffect(MODELS_DIR, _source.GetBaseFrame(), (ArtifactReductionEffectMode)cbArtifactReductionMode.SelectedIndex);
                        _videoEffect.Init(info.Width, info.Height);
                    }
                    
                    break;
                case 2:
                    {
                        _videoEffect = new SuperResolutionEffect(MODELS_DIR, _source.GetBaseFrame(), (SuperResolutionEffectMode)cbSuperResolutionMode.SelectedIndex, Convert.ToInt32(edSuperResolutionHeight.Text));
                        _videoEffect.Init(info.Width, info.Height);
                    }
                    
                    break;
                case 3:
                    {
                        _videoEffect = new UpscaleEffect(MODELS_DIR, _source.GetBaseFrame(), (float)(slUpscaleStrength.Value / 10.0), Convert.ToInt32(edUpscaleHeight.Text));
                        _videoEffect.Init(info.Width, info.Height);
                    }
                    
                    break;
                case 4:
                    {
                        var eff = new AIGSEffect(MODELS_DIR, _source.GetBaseFrame(), (AIGSEffectMode)cbAIGSMode.SelectedIndex);
                        eff.BackgroundImage = edAIGSBackground.Text;

                        _videoEffect = eff;
                        _videoEffect.Init(info.Width, info.Height);
                    }
                    break;
                default:
                    break;
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

            var processedFrame = _videoEffect.Process();
            _output.WriteFrame(processedFrame);

            //OpenCvSharp.Cv2.ImWrite("c:\\vf\\x\\proc.jpg", processedFrame);

            if (_totalFrames > 0)
            {
                var progress = (int)((_frameID * 100) / _totalFrames);
                if (progress != _lastProgress)
                {
                    _lastProgress = progress;
                    UpdateProgress(progress);
                }
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
            _stopFlag = true;

            Thread.Sleep(250);

            _source?.Stop();
            _source?.Dispose();
            _source = null;

            _output?.Finish();
            _output?.Dispose();
            _output = null;

            _videoEffect?.Dispose();
            _videoEffect = null;

            Dispatcher.Invoke(() =>
            {
                pbProgress.Value = 0;

                pnScreen.BeginInit();
                pnScreen.Source = null;
                pnScreen.EndInit();
            });
        }

        private void RenderFrame(OpenCvSharp.Mat frame)
        {
            if (_stopFlag)
            {
                return;
            }

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

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            var list = CameraVideoSource.ListDevices();
            foreach (var item in list)
            {
                cbCamera.Items.Add(item);
            }

            if (cbCamera.Items.Count > 0)
            {
                cbCamera.SelectedIndex = 0;
            }
        }

        private void cbEffect_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (gdDenoise == null)
            {
                return;
            }

            switch (cbEffect.SelectedIndex)
            {
                case 0:
                    {
                        gdDenoise.Visibility = Visibility.Visible;
                        gdArtifactReduction.Visibility = Visibility.Collapsed;
                        gdSuperResolution.Visibility = Visibility.Collapsed;
                        gdUpscale.Visibility = Visibility.Collapsed;
                        gdAIGS.Visibility = Visibility.Collapsed;
                    }
                    break;
                case 1:
                    {
                        gdDenoise.Visibility = Visibility.Collapsed;
                        gdArtifactReduction.Visibility = Visibility.Visible;
                        gdSuperResolution.Visibility = Visibility.Collapsed;
                        gdUpscale.Visibility = Visibility.Collapsed;
                        gdAIGS.Visibility = Visibility.Collapsed;
                    }
                    break;
                case 2:
                    {
                        gdDenoise.Visibility = Visibility.Collapsed;
                        gdArtifactReduction.Visibility = Visibility.Collapsed;
                        gdSuperResolution.Visibility = Visibility.Visible;
                        gdUpscale.Visibility = Visibility.Collapsed;
                        gdAIGS.Visibility = Visibility.Collapsed;
                    }
                    break;
                case 3:
                    {
                        gdDenoise.Visibility = Visibility.Collapsed;
                        gdArtifactReduction.Visibility = Visibility.Collapsed;
                        gdSuperResolution.Visibility = Visibility.Collapsed;
                        gdUpscale.Visibility = Visibility.Visible;
                        gdAIGS.Visibility = Visibility.Collapsed;
                    }
                    break;
                case 4:
                    {
                        gdDenoise.Visibility = Visibility.Collapsed;
                        gdArtifactReduction.Visibility = Visibility.Collapsed;
                        gdSuperResolution.Visibility = Visibility.Collapsed;
                        gdUpscale.Visibility = Visibility.Collapsed;
                        gdAIGS.Visibility = Visibility.Visible;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
}
