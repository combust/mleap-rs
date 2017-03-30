using System;

namespace MLeapDotNet
{
    public class Frame : IDisposable
    {
        private readonly int _rowsCount;
        private readonly IntPtr _frame;
        private bool _isDisposed;

        public Frame(int rowsCount)
        {
            _rowsCount = rowsCount;
            _frame = NativeMethods.mleap_frame_with_size((uint) _rowsCount);
        }

        internal IntPtr NativePointer => _frame;

        public int RowsCount => _rowsCount;

        public void AddDoubles(string name, params double[] values)
        {
            if (values.Length != _rowsCount)
            {
                throw new InvalidOperationException("Row count mismatch");
            }
            NativeMethods.mleap_frame_with_doubles(_frame, name, values);
        }

        public void AddStrings(string name, params string[] values)
        {
            if (values.Length != _rowsCount)
            {
                throw new InvalidOperationException("Row count mismatch");
            }
            NativeMethods.mleap_frame_with_strings(_frame, name, values);
        }

        public double[] GetDoubles(string name)
        {
            var result = new double[_rowsCount];
            NativeMethods.mleap_frame_get_doubles(_frame, name, result);
            return result;
        }

        ~Frame()
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
            NativeMethods.mleap_frame_free(_frame);
            _isDisposed = true;
        }
    }
}
