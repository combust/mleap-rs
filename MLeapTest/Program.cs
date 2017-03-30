using System;
using MLeapDotNet;

namespace MLeapTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // NOTE: to use the non-extended API, remove the frame transform callback 
            using (var transformer = Transformer.LoadFrom("transformer.mleap",
                f =>
                {
                    f.AddDoubles("price_prediction", new double[f.RowsCount]);
                }))
            using (var frame = new Frame(2))
            {
                frame.AddDoubles("bathrooms", 2.0, 1.0);
                frame.AddDoubles("bedrooms", 3.0, 2.0);
                frame.AddDoubles("security_deposit", 50.0, 50.0);
                frame.AddDoubles("cleaning_fee", 30.0, 30.0);
                frame.AddDoubles("extra_people", 0.0, 0.0);
                frame.AddDoubles("number_of_reviews", 23.0, 15.0);
                frame.AddDoubles("square_feet", 1200.0, 900.0);
                frame.AddDoubles("review_scores_rating", 93.0, 90.0);

                frame.AddStrings("cancellation_policy", "strict", "strict");
                frame.AddStrings("host_is_superhost", "1.0", "1.0");
                frame.AddStrings("instant_bookable", "1.0", "1.0");
                frame.AddStrings("room_type", "Entire home/apt", "Entire home/apt");
                frame.AddStrings("state", "NY", "NY");

                transformer.Transfrom(frame);

                var result = frame.GetDoubles("price_prediction");
                if (result[0] != 236.76099900182078)
                {
                    throw new Exception("fail");
                }
            }
        }
    }
}
