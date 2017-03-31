using System;
using System.Collections.Generic;

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
            _frame = NativeMethods.mleap_frame_with_size((UIntPtr) _rowsCount);
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

        public IEnumerable<DoubleTensor> GetTensors(string name)
        {
            for (int i = 0; i < _rowsCount; i++)
            {
                UIntPtr dimensionsLength;
                UIntPtr valuesLength;
                NativeMethods.mleap_frame_get_double_tensor_len(_frame, name, (UIntPtr) i, out dimensionsLength,
                    out valuesLength);
                var dimensions = new UIntPtr[(ulong) dimensionsLength];
                var values = new double[(ulong) valuesLength];
                NativeMethods.mleap_frame_get_double_tensor(_frame, name, (UIntPtr) i, dimensions, values);
                yield return new DoubleTensor(new UIntPtrArrayAdapter(dimensions), values);
            }
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
