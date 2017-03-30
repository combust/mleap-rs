using System;
using System.Collections.Generic;

namespace MLeapDotNet
{
    public class Transformer : IDisposable
    {
        private readonly IntPtr _transformer;
        private readonly Dictionary<IntPtr, Frame> _currentFrames = new Dictionary<IntPtr, Frame>(); // not thread safe
        private bool _isDisposed;

        private Transformer(string modelDirectoryPath, Action<Frame> transform)
        {
            _transformer = transform == null
                ? NativeMethods.mleap_transformer_load(modelDirectoryPath)
                : NativeMethods.mleap_transformer_load_ex(modelDirectoryPath,
                    f => transform(_currentFrames[f]));
        }

        public static Transformer LoadFrom(string modelDirectoryPath)
        {
            return new Transformer(modelDirectoryPath, null);
        }

        public static Transformer LoadFrom(string modelDirectoryPath, Action<Frame> transform)
        {
            return new Transformer(modelDirectoryPath, transform);
        }

        public void Transfrom(Frame frame)
        {
            _currentFrames[frame.NativePointer] = frame;
            NativeMethods.mleap_transform(_transformer, frame.NativePointer);
            _currentFrames.Remove(frame.NativePointer);
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
