using NvidiaMaxine.VideoEffects;
using OpenCvSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection.PortableExecutable;

namespace DenoiseEffectApp
{
    internal class Program
    {
        private static Context Context;
        //#define DEFAULT_CODEC "H264"

        static void Main(string[] args)
        {
            Err fxErr = Err.errNone;
            int nErrs = 0;
            FXApp app = new FXApp();

            //nErrs = ParseMyArgs(argc, argv);
            //if (nErrs)
            //    std::cerr << nErrs << " command line syntax problems\n";

            if (Context.Webcam)
            {
                // If webcam is on, enable showing the results and turn off displaying the progress
                if (Context.Progress)
                {
                    Context.Progress = !Context.Progress;
                }

                if (!Context.Show)
                {
                    Context.Show = !Context.Show;
                }

            }
            if (string.IsNullOrEmpty(Context.InFile) && !Context.Webcam)
            {
                Console.WriteLine("Please specify --in_file=XXX or --webcam=true");
                ++nErrs;
            }
            if (string.IsNullOrEmpty(Context.OutFile) && !Context.Show)
            {
                Console.WriteLine("Please specify --out_file=XXX or --show");
                ++nErrs;
            }

            app._progress = Context.Progress;
            app.setShow(Context.Show);

            if (nErrs > 0)
            {
                Usage();
                fxErr = Err.errFlag;
            }
            else
            {
                fxErr = app.createEffect(NvVFXFilterSelectors.NVVFX_FX_DENOISING, Context.ModelDir);
                if (Err.errNone != fxErr)
                {
                    Console.WriteLine("Error creating effect");
                }
                else
                {
                    if (Helpers.IsImageFile(Context.InFile))
                        fxErr = app.processImage(Context.InFile, Context.OutFile, Context.Strength);
                    else
                        fxErr = app.processMovie(Context);
                }
            }

            if (fxErr > 0)
            {
                Console.WriteLine("Error: " + app.errorStringFromCode(fxErr));
            }
        }
        static void Usage()
        {
            Console.WriteLine("DenoiseEffectApp [args ...]\n" +
              "  where args is:\n" +
              "  --in_file=<path>           input file to be processed (can be an image but the best denoising performance is observed on videos)\n" +
              "  --webcam                   use a webcam as the input\n" +
              "  --out_file=<path>          output file to be written\n" +
              "  --show                     display the results in a window (for webcam, it is always true)\n" +
              "  --strength=<value>         strength of an effect [0-1]\n" +
              "  --model_dir=<path>         the path to the directory that contains the models\n" +
              "  --codec=<fourcc>           the fourcc code for the desired codec (default " + Context.DEFAULT_CODEC + ")\n" +
              "  --progress                 show progress\n" +
              "  --verbose                  verbose output\n" +
              "  --debug                    print extra debugging information\n"
            );
        }
    }
}