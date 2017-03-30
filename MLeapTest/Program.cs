using MLeap;

namespace MLeapTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var transformer = NativeMethods.mleap_transformer_load(@"transformer.mleap");
            try
            {
                var frame = NativeMethods.mleap_frame_with_size(1);
                try
                {
                    NativeMethods.mleap_frame_with_doubles(frame, "bathrooms", new[] { 2.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "bedrooms", new[] { 3.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "security_deposit", new[] { 50.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "cleaning_fee", new[] { 30.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "extra_people", new[] { 0.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "number_of_reviews", new[] { 23.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "square_feet", new[] { 1200.0 });
                    NativeMethods.mleap_frame_with_doubles(frame, "review_scores_rating", new[] { 93.0 });

                    NativeMethods.mleap_frame_with_strings(frame, "cancellation_policy", new[] { "strict" });
                    NativeMethods.mleap_frame_with_strings(frame, "host_is_superhost", new[] { "1.0" });
                    NativeMethods.mleap_frame_with_strings(frame, "instant_bookable", new[] { "1.0" });
                    NativeMethods.mleap_frame_with_strings(frame, "room_type", new[] { "Entire home/apt" });
                    NativeMethods.mleap_frame_with_strings(frame, "state", new[] { "NY" });

                    NativeMethods.mleap_transform(transformer, frame);

                    var result = new double[1];
                    NativeMethods.mleap_frame_get_doubles(frame, "price_prediction", result);
                }
                finally
                {
                    NativeMethods.mleap_frame_free(frame);
                }
            }
            finally
            {
                NativeMethods.mleap_transformer_free(transformer);
            }
        }
    }
}
