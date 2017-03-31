using System;

namespace MLeapDotNet
{
    public class Model
    {
        private readonly IntPtr _model;

        public Model(IntPtr model)
        {
            _model = model;
        }

        public double GetDouble(string name)
        {
            return NativeMethods.mleap_model_get_double(_model, name);
        }

        public DoubleTensor GetDoubleTensor(string name)
        {
            UIntPtr dimensionsLength;
            UIntPtr valuesLength;
            NativeMethods.mleap_model_get_double_tensor_len(_model, name, out dimensionsLength, out valuesLength);
            var dimensions = new UIntPtr[(int) dimensionsLength];
            var values = new double[(int) valuesLength];
            NativeMethods.mleap_model_get_double_tensor(_model, name, dimensions, values);
            return new DoubleTensor(new UIntPtrArrayAdapter(dimensions), values);
        }
    }
}
