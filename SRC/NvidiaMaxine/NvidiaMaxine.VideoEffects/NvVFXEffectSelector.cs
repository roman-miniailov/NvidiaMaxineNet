using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects
{
    public class NvVFXEffectSelector 
    {
        public string Name { get; set; }

        public NvVFXEffectSelector(string name)
        {
            Name = name;
        }
    }
}
