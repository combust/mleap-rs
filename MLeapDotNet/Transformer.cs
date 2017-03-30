using System;

namespace MLeapDotNet
{
    public class Transformer : IDisposable
    {
        private readonly IntPtr _transformer;
        private bool _isDisposed;

        private Transformer(string modelDirectoryPath)
        {
            _transformer = NativeMethods.mleap_transformer_load(modelDirectoryPath);
        }

        public static Transformer LoadFrom(string modelDirectoryPath)
        {
            return new Transformer(modelDirectoryPath);
        }

        public void Transfrom(Frame frame)
        {
            NativeMethods.mleap_transform(_transformer, frame.NativePointer);
        }

        ~Transformer()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
        }

        private void Dispose(bool isDisposing)
        {
            if (_isDisposed)
            {
                return;
            }
            NativeMethods.mleap_transformer_free(_transformer);
            _isDisposed = true;
        }
    }
}
