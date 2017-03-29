using MLeap;

namespace MLeapTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var frame = NativeMethods.mleap_frame_with_size(42);
            NativeMethods.mleap_frame_with_doubles(frame, "x", new[] { 1.0d });
            NativeMethods.mleap_frame_free(frame);
        }
    }
}
