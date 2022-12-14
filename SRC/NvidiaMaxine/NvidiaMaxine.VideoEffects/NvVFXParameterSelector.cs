using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public class NvVFXParameterSelector
    {
        public string Name { get; set; }

        public NvVFXParameterSelector(string name)
        {
            Name = name;
        }
    }
}
