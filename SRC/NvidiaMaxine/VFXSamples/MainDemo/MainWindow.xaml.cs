using NvidiaMaxine.VideoEffects;
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

        private ulong _frameID;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void btStart_Click(object sender, RoutedEventArgs e)
        {
            _frameID = 0;

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

                _source.GetVideoInfo(out info);
            }

            _output = new FileOutput();
            _output.Init(edOutputFilename.Text, info.Resolution, info.FrameRate);

            _source.Start();
        }

        private void Source_FrameReady(object sender, VideoFrameEventArgs e)
        {
            Debug.WriteLine("Frame received.");

            _output.WriteFrame(e.Frame);

            _frameID++;
        }

        private void btStop_Click(object sender, RoutedEventArgs e)
        {
            _source?.Stop();
            _source?.Dispose();
            _source = null;

            _output?.Finish();
            _output?.Dispose();
            _output = null;
        }
    }
}
