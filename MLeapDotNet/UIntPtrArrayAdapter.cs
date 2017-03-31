using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace MLeapDotNet
{
    internal class UIntPtrArrayAdapter : IReadOnlyList<ulong>
    {
        private readonly UIntPtr[] _array;

        public UIntPtrArrayAdapter(UIntPtr[] array)
        {
            _array = array;
        }

        public int Count => _array.Length;

        public ulong this[int index] => (ulong)_array[index];

        public IEnumerator<ulong> GetEnumerator()
        {
            return _array.Select(v => (ulong) v).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
