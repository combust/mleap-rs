using System.Collections.Generic;

namespace MLeapDotNet
{
    public class DoubleTensor
    {
        public DoubleTensor(IReadOnlyList<ulong> dimensions, double[] values)
        {
            Dimensions = dimensions;
            Values = values;
        }

        public IReadOnlyList<ulong> Dimensions { get; }

        public double[] Values { get; }
    }
}
