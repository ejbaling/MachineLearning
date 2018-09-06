using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace mlApp
{
    class Program
    {
        public class ProductData
        {
            [Column(ordinal: "0")]
            public float ProductId;

            [Column(ordinal: "1")]
            public float Year;

            [Column(ordinal: "2")]
            public float Month;

            [Column(ordinal: "3")]
            public float SalesCount;

            [Column(ordinal: "4")]
            public float MinQuantity;

            [Column(ordinal: "5")]
            public float MaxQuantity;

            [Column(ordinal: "6")]
            public float Quantity;

            [Column(ordinal: "7")]
            public float Prev;

            [Column(ordinal: "8", name: "Label")]
            public float Next;
        }

        // STEP 1: Define your data structures

        // WeatherData is used to provide training data, and as 
        // input for prediction operations
        // - First 4 properties are inputs/features used to predict the label
        // - Label is what you are predicting, and is only set when training
        public class WeatherData
        {
            [Column("0")]
            public float Temperature;

            [Column("1")]
            public float Humidity;

            [Column("2")]
            public float MinTemp;

            [Column("3")]
            public float MaxTemp;

            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        public class ProductUnitPrediction
        {
            public float Score;
        }

        static void Main(string[] args)
        {
            // STEP 2: Create a pipeline and load your data
            var pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of weather-data.txt is set to 'Copy always'
            string dataPath = "product-data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<ProductData>(true, separator: ','));

            // The model needs the columns to be arranged into a single column of numeric type
            // First, we group all numeric columns into a single array named NumericalFeatures
            pipeline.Add(new ColumnConcatenator("NumericalFeatures", nameof(ProductData.Year), 
                nameof(ProductData.Month),
                nameof(ProductData.SalesCount),
                nameof(ProductData.MinQuantity),
                nameof(ProductData.MaxQuantity),
                nameof(ProductData.Quantity),
                nameof(ProductData.Prev)
            ));

            // Second group is for categorical features (just one in this case), we name this column CategoryFeatures
            pipeline.Add(new ColumnConcatenator("CategoryFeatures", nameof(ProductData.ProductId)));

            // Then we need to transform the category column using one-hot encoding. This will return a numeric array
            pipeline.Add(new CategoricalOneHotVectorizer("CategoryFeatures"));

            // Once all columns are numeric types, all columns will be combined
            // into a single column, named Features 
            pipeline.Add(new ColumnConcatenator("Features", "NumericalFeatures", "CategoryFeatures"));

            // Add the Learner to the pipeline. The Learner is the machine learning algorithm used to train a model
            // In this case, TweedieFastTree.TrainRegression was one of the best performing algorithms, but you can 
            // choose any other regression algorithm (StochasticDualCoordinateAscentRegressor,PoissonRegressor,...)
            pipeline.Add(new FastTreeTweedieRegressor { NumThreads = 1, FeatureColumn = "Features" });

            // Convert the Label back into original text (after converting to number in step 3)
            //pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            // Finally, we train the pipeline using the training dataset set at the first stage
            var model = pipeline.Train<ProductData, ProductUnitPrediction>();

            var prediction = model.Predict(new ProductData()
            {
                ProductId = 895,
                Year = 2017,
                Month = 9,
                SalesCount = 39,
                MinQuantity = 1,
                MaxQuantity = 2,
                Quantity = 46,
                Prev = 0
            });
            Console.WriteLine($"Predicted quantity is: {prediction.Score}");

            Console.ReadLine();
        }
    }
}
