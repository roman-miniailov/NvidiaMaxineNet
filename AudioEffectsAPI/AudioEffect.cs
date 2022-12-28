//using NAudio.Wave;
//using NvidiaMaxine.AudioEffects.API;
//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;

//namespace NvidiaMaxine.AudioEffects
//{
//    public class AudioEffect
//    {

//        bool generate_output(
//            IntPtr handle_,
//            string input_wav,
//            uint input_sample_rate,
//            int num_input_samples_per_frame,
//            bool is_aec,
//            string input_farend_wav,
//            string output_wav,
//            uint output_sample_rate,
//            uint num_output_channels,
//            uint num_output_samples_per_frame)
//        {
//            float[] audio_data = AudioFileReader.ReadWAVFile(input_wav, input_sample_rate, num_input_samples_per_frame);

//            if (audio_data == null)
//            {
//                Debug.WriteLine("Unable to read wav file: " + input_wav);
//                return false;
//            }

//            Debug.WriteLine("Input wav file: " + input_wav);
//            Debug.WriteLine($"Total {audio_data.Length} samples read");

//            float[] farend_audio_data = null;
//            if (is_aec)
//            {
//                farend_audio_data = AudioFileReader.ReadWAVFile(input_farend_wav, input_sample_rate, num_input_samples_per_frame);
//                if (farend_audio_data == null)
//                {
//                    Debug.WriteLine("Unable to read wav file: " + input_farend_wav);
//                    return false;
//                }

//                Debug.WriteLine($"Input wav file: {input_farend_wav}");
//                Debug.WriteLine($"Total {farend_audio_data.Length} samples read");
//            }

//            var wav_write = new AudioFileWriter(output_wav, output_sample_rate, num_output_channels, 32, true);

//            float frame_in_secs = (float)num_input_samples_per_frame / (float)input_sample_rate;
//            TimeSpan total_run_time = TimeSpan.Zero;
//            float total_audio_duration = 0.0f;
//            float checkpoint = 0.1f;
//            float expected_audio_duration = (float)audio_data.Length / (float)input_sample_rate;
//            var frame = new float[num_output_samples_per_frame];

//            //std::string progress_bar = "[          ] ";
//            //std::cout << "Processed: " << progress_bar << "0%\r";
//            //std::cout.flush();

//            var final_audio_size = audio_data.Length;
            
//            //Taking the min size of farend and nearend if their sizes mismatch
//            if (is_aec)
//            {
//                if (audio_data.Length != farend_audio_data.Length)
//                {
//                    final_audio_size = Math.Min(audio_data.Length, farend_audio_data.Length);
//                }
//            }
            
//            // wav data is already padded to align to num_samples_per_frame by ReadWavFile()
//            for (int offset = 0; offset < final_audio_size; offset += num_input_samples_per_frame)
//            {
//                Stopwatch stopwatch = new Stopwatch();
//                stopwatch.Start();
//                if (is_aec)
//                {
//                    const float* input[2];
//                    float* output[1];
//                    input[0] = &audio_data.data()[offset];
//                    input[1] = &farend_audio_data.data()[offset];
//                    output[0] = frame.get();
//                    if (NvAFXAPI.NvAFX_Run(handle_, input, output, num_input_samples_per_frame_, num_input_channels_) != NVAFX_STATUS_SUCCESS)
//                    {
//                        std::cerr << "NvAFX_Run() failed" << std::endl;
//                        return false;
//                    }
//                }
//                else
//                {
//                    const float* input[1];
//                    float* output[1];
//                    input[0] = &audio_data.data()[offset];
//                    output[0] = frame.get();
//                    if (NvAFXAPI.NvAFX_Run(handle_, input, output, num_input_samples_per_frame_, num_input_channels_) != NVAFX_STATUS_SUCCESS)
//                    {
//                        std::cerr << "NvAFX_Run() failed" << std::endl;
//                        return false;
//                    }
//                }

//                stopwatch.Stop();
//                total_run_time += (std::chrono::duration<float>(run_end_tick - start_tick)).count();
//                total_audio_duration += frame_in_secs;

//                if ((total_audio_duration / expected_audio_duration) >= checkpoint)
//                {
//                    progress_bar[(int)(checkpoint * 10.0f)] = '=';
//                    std::cout << "Processed: " << progress_bar << checkpoint * 100.f << "%" << (checkpoint >= 1 ? "\n" : "\r");
//                    std::cout.flush();
//                    checkpoint += 0.1f;
//                }


//                wav_write.writeChunk(frame.get(), num_output_samples_per_frame_ * sizeof(float));

//                if (real_time_)
//                {
//                    auto end_tick = std::chrono::high_resolution_clock::now();
//                    std::chrono::duration<float> elapsed = end_tick - start_tick;
//                    float sleep_time_secs = frame_in_secs - elapsed.count();
//                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time_secs * 1000)));
//                }
//            }

//            std::cout << "Processing time " << std::setprecision(2) << total_run_time
//                      << " secs for " << total_audio_duration << std::setprecision(2)
//                      << " secs audio file (" << total_run_time / total_audio_duration
//                      << " secs processing time per sec of audio)" << std::endl;

//            if (real_time_)
//            {
//                std::cout << "Note: App ran in real time mode i.e. simulated the input data rate of a mic" << std::endl
//                          << "'Processing time' could be less then actual run time" << std::endl;
//            }

//            wav_write.commitFile();

//            std::cout << "Output wav file written. " << output_wav << std::endl
//                      << "Total " << audio_data.size() << " samples written"
//                      << std::endl;

//            if (NvAFX_DestroyEffect(handle_) != NVAFX_STATUS_SUCCESS)
//            {
//                std::cerr << "NvAFX_DestroyEffect() failed" << std::endl;
//                return false;
//            }

//            return true;
//        }
//    }
//}
