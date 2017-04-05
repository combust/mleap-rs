using System;
using System.Runtime.InteropServices;

namespace MLeapDotNet
{
    public static class NativeMethods
    {
        // NOTE: When creating wrappers, use UIntPtr as C# equivalent of Rust's usize

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate IntPtr LoadModelDelegate(IntPtr c_model);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void TransformDelegate(IntPtr c_frame, IntPtr c_model);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr mleap_frame_with_size(UIntPtr c_size);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl)]
        public static extern void mleap_frame_free(IntPtr c_frame);
        
        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_with_doubles(IntPtr c_frame, string c_name, double[] c_values);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_with_strings(IntPtr c_frame, string c_name, string[] c_values);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr mleap_transformer_load(string c_path);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr mleap_transformer_load_ex(string c_path, LoadModelDelegate c_load_model,
            TransformDelegate c_transform);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl)]
        public static extern void mleap_transformer_free(IntPtr c_transformer);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl)]
        public static extern void mleap_transform(IntPtr c_transformer, IntPtr c_frame);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_get_doubles(IntPtr c_frame, string c_name, double[] c_buffer);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern UIntPtr mleap_frame_get_double_tensors_len(IntPtr c_frame, string c_name);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_get_double_tensor_len(IntPtr c_frame, string c_name, UIntPtr index,
            out UIntPtr c_dimensions_len, out UIntPtr c_values_len);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_frame_get_double_tensor(IntPtr c_frame, string c_name, UIntPtr index,
            UIntPtr[] c_dimensions, double[] c_values);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern double mleap_model_get_double(IntPtr c_model, string c_name);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_model_get_double_tensor_len(IntPtr c_model, string c_name,
            out UIntPtr c_dimensions_len, out UIntPtr c_values_len);

        [DllImport("mleap", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern void mleap_model_get_double_tensor(IntPtr c_model, string c_name,
            UIntPtr[] c_dimensions, double[] c_values);
    }
}
