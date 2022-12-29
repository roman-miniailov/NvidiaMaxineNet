using NvidiaMaxine.AudioEffects;
using NvidiaMaxine.AudioEffects.Effects;
using NvidiaMaxine.AudioEffects.Outputs;
using NvidiaMaxine.AudioEffects.Sources;
using System;
using System.Diagnostics;
using System.Windows;

namespace MainDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private BaseEffect _effect;

        private IAudioSource _source;

        private WAVFileWriter _fileWriter;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void btSelectSourceFile_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.OpenFileDialog();
            dlg.DefaultExt = ".wav";
            dlg.Filter = "WAV Files (*.wav)|*.wav";
            if (dlg.ShowDialog() == true)
            {
                edSourceFile.Text = dlg.FileName;
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            var audioSources = AudioCaptureSource.Enumerate();
            foreach (var audioSource in audioSources)
            {
                cbSourceDevice.Items.Add(audioSource);
            }

            if (cbSourceDevice.Items.Count > 0)
            {
                cbSourceDevice.SelectedIndex = 0;
            }
        }

        private void btSelectOutputFile_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new Microsoft.Win32.SaveFileDialog();
            dlg.Filter = "WAV Files (*.wav)|*.wav";
            if (dlg.ShowDialog() == true)
            {
                edOutputFile.Text = dlg.FileName;
            }
        }

        private void btStart_Click(object sender, RoutedEventArgs e)
        {
            // create effect
            switch (cbEffect.SelectedIndex)
            {
                case 0:
                    _effect = new DenoiserEffect(edModelsFolder.Text, SampleRate.SR48000);
                    break;
                case 1:
                    _effect = new DereverbEffect(edModelsFolder.Text, SampleRate.SR48000);
                    break;
                default:
                    throw new ArgumentOutOfRangeException("Wrong effect index");
            }            
            
            if (!_effect.Init())
            {
                MessageBox.Show("Failed to initialize effect");
                return;
            }

            // create source
            if (rbSourceFile.IsChecked == true)
            {
                _source = new WAVFileSource(edSourceFile.Text, (int)_effect.SampleRate, 1, 16);
                _source.DataAvailable += _source_DataAvailable;
                _source.Complete += _source_Complete;                
            }
            else
            {
                _source = new AudioCaptureSource(cbSourceDevice.Text, (int)_effect.SampleRate, 1, 16, true);
                _source.DataAvailable += _source_DataAvailable;
                _source.Complete += _source_Complete;
            }
            
            // create output
            if (rbOutputFile.IsChecked == true)
            {
                _fileWriter = new WAVFileWriter(edOutputFile.Text, (int)_effect.SampleRate, 1, 32, true);
            }

            _source.Start((int)_effect.FrameSize);
        }

        private void _source_Complete(object sender, EventArgs e)
        {
            _fileWriter?.Finish();
            _fileWriter?.Dispose();
            
            MessageBox.Show("Complete");
        }

        private void _source_DataAvailable(object sender, float[] e)
        {
            // process
            var res = _effect.Process(e, 0, e.Length);
            if (!res)
            {
                Debug.WriteLine("Failed to process audio");
                return;
            }

            _fileWriter?.Write(e);
        }

        private void btStop_Click(object sender, RoutedEventArgs e)
        {
            _source.Stop();
        }

        private void btSelectModelsFolder_Click(object sender, RoutedEventArgs e)
        {
            var fb = new System.Windows.Forms.FolderBrowserDialog();
            if (fb.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                edModelsFolder.Text = fb.SelectedPath;
            }
        }
    }
}
