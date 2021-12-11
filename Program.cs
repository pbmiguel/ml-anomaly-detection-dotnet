using System;
using System.IO;
using System.Linq;
using Microsoft.ML;

namespace ml_anomaly_detection
{
    public class Program
    {
        public static void Main(string[] args)
        {
            /*
             * Anomaly Detection Using ML.NET
             * https://www.infoq.com/articles/anomaly-detection-ml-net/?topicPageSponsorship=8e4bb459-3843-4916-b7a8-0fbd274b1496&itm_source=articles_about_ai-ml-data-eng&itm_medium=link&itm_campaign=ai-ml-data-eng
             */
            var trainingSetPath = "data/train/*";
            var testingSetPath = "data/test/*";
            var ml = new MLContext();

            var trainingDataView = ml.Data.LoadFromTextFile<Features>(trainingSetPath, hasHeader: true, separatorChar: ',');
            var testingDataView = ml.Data.LoadFromTextFile<Features>(testingSetPath, hasHeader: true, separatorChar: ',');

            /**
             * Now create a training pipeline. 
             * Here you select Anomaly Detection trainer in the form of Randomized PCA, to which you define the names of the features in the parameters.
             * Additionally, you can set up some options like Rank (the number of components in the PCA) or Seed (The seed for random number generation).
             */
            var columnNames = new[]   
            {
                "Xposition", "Yposition", "Zposition", "FirstSensorActivity", "SecondSensorActivity", "ThirdSensorActivity", "FourthSensorActivity", "Anomaly"
            };
            var options = new Microsoft.ML.Trainers.RandomizedPcaTrainer.Options()
            {
                Rank = 4
            };
            var pipeline = ml.Transforms.Concatenate("Features", columnNames)
                .Append(ml.AnomalyDetection.Trainers.RandomizedPca(options));

            /*
             * Finally, you can move on to training and testing the model, which is limited to three lines of code.
             */
            var model = pipeline.Fit(trainingDataView);
            var predictions = model.Transform(testingDataView);

            var results = ml.Data.CreateEnumerable<Result>(predictions, reuseRowObject: false).ToList();
            
            /*
             * As you may have noticed before, the Result class is used to capture prediction.
             * PredictedLabel determines whether it is an outlier (true) or an inhaler (false).
             * Score is the result of the anomaly and a data point with a predicted score higher than 0.5 is usually considered an outlier.
             * You can display the outliers in the console using the following code:
             */
            foreach (var result in results.Where(result => result.PredictedLabel))
            {
                Console.WriteLine("The example is an outlier with a score of being outlier {0}", result.Score);
            }
        }
    }
}