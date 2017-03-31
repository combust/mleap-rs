using System;
using System.Collections.Generic;

namespace MLeapDotNet
{
    public class Transformer
    {
        public static Transformer<TModel> LoadFrom<TModel>(string modelDirectoryPath, Func<Model, TModel> loadModel,
            Action<Frame, TModel> transform)
        {
            return new Transformer<TModel>(modelDirectoryPath, loadModel, transform);
        }
    }

    public class Transformer<TModel> : IDisposable
    {
        private readonly TModel _model;
        private readonly IntPtr _transformer;
        private readonly Dictionary<IntPtr, Frame> _currentFrames = new Dictionary<IntPtr, Frame>(); // not thread safe
        private bool _isDisposed;

        internal Transformer(string modelDirectoryPath, Func<Model, TModel> loadModel, Action<Frame, TModel> transform)
        {
            var model = default(TModel);
            _transformer = transform == null
                ? NativeMethods.mleap_transformer_load(modelDirectoryPath)
                : NativeMethods.mleap_transformer_load_ex(modelDirectoryPath,
                    m =>
                    {
                        model = loadModel(new Model(m));
                        return IntPtr.Zero;
                    },
                    (f, m) => transform(_currentFrames[f], _model));
            _model = model;
        }

        public static Transformer<bool> LoadFrom(string modelDirectoryPath)
        {
            return new Transformer<bool>(modelDirectoryPath, null, null);
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
