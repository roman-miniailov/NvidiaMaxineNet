using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NvidiaMaxine.VideoEffects.Effects
{
    public class EffectsPipeline
    {
        private List<BaseEffect> _effects = new List<BaseEffect>();

        public void AddEffect(BaseEffect effect)
        {
            _effects.Add(effect);
        }

        public void RemoveEffect(BaseEffect effect)
        {
            _effects.Remove(effect);
        }

        public void ClearEFfects()
        {
            _effects.Clear();
        }

        public void PushFrame(Mat frame)
        {
            
        }
        
        public bool Init(VideoInfo info)
        {
            return true;
        }
    }
}
