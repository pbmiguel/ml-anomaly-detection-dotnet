using Microsoft.ML.Data;

namespace ml_anomaly_detection
{
    /**
     * This is a schema class of the data in /data
     */
    public class Features
    {
        [LoadColumn(0)] 
        public float Xposition { get; set; }

        [LoadColumn(1)] 
        public float Yposition { get; set; }

        [LoadColumn(2)] 
        public float Zposition { get; set; }

        [LoadColumn(3)] 
        public float FirstSensorActivity { get; set; }

        [LoadColumn(4)] 
        public float SecondSensorActivity { get; set; }
        
        [LoadColumn(5)] 
        public float ThirdSensorActivity { get; set; }
        
        [LoadColumn(6)] 
        public float FourthSensorActivity { get; set; }
        
        [LoadColumn(7)] 
        public float Anomaly { get; set; }
    }
}