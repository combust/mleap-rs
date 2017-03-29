using System;
using System.Runtime.InteropServices;

namespace MLeap
{
    public static class NativeMethods
    {
        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mleap_frame_with_size(uint c_size);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl)]
        public static extern void mleap_frame_free(IntPtr c_frame);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_with_doubles(IntPtr c_frame, string c_name, double[] c_values);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_with_strings(IntPtr c_frame, string c_name, string[] c_values);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr mleap_transformer_load(string c_path);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_transformer_free(IntPtr c_transformer);

        [DllImport("mleap-rs", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_transform(IntPtr c_transformer, IntPtr c_frame);
    }
}
