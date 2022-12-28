using System.Runtime.InteropServices;

namespace CUDA {
	/// <summary>
	/// (FP16) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct __half {
		public ushort x;
	}

	/// <summary>
	/// (FP16) 
	/// </summary>
	[StructLayout(LayoutKind.Sequential)]
	public struct __half2 {
		public uint x;
	}
}
