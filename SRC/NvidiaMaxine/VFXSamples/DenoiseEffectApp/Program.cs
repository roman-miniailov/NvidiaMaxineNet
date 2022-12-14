namespace DenoiseEffectApp
{
    internal class Program
    {
        private static string DEFAULT_CODEC = "avc1";
        //#define DEFAULT_CODEC "H264"

        static void Main(string[] args)
        {
            bool FLAG_debug = false;
            bool FLAG_verbose = false;
            bool FLAG_show = false;
            bool FLAG_progress = false;
            bool FLAG_webcam = false;
            float FLAG_strength = 0.0f;
            string FLAG_codec = DEFAULT_CODEC;
            string FLAG_camRes = "1280x720";
            string FLAG_inFile;
            string FLAG_outFile;
            string FLAG_outDir;
            string FLAG_modelDir;


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
              "  --codec=<fourcc>           the fourcc code for the desired codec (default " + DEFAULT_CODEC + ")\n" +
              "  --progress                 show progress\n" +
              "  --verbose                  verbose output\n" +
              "  --debug                    print extra debugging information\n"
            );
        }
    }
}