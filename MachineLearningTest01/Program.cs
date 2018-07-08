using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearningTest01
{
	class Program
	{
		public class IrisData
		{
			[Column("0")]
			public float SepalLength;

			[Column("1")]
			public float SepalWidth;

			[Column("2")]
			public float PetalLength;

			[Column("3")]
			public float PetalWidth;

			[Column("4")]
			[ColumnName("Label")]
			public string Label;

		}

		public class IrisPrediction
		{
			[ColumnName("PredictedLabel")]
			public string PredictedLabels;
		}

		static void Main(string[] args)
		{
			var pipeLine = new LearningPipeline();

			var dataPath = "IrisData.txt";

			pipeLine.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));
			pipeLine.Add(new Dictionarizer("Label"));
			pipeLine.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
			pipeLine.Add(new StochasticDualCoordinateAscentClassifier());
			pipeLine.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

			var model = pipeLine.Train<IrisData, IrisPrediction>();
			var prediction = model.Predict(new IrisData()
			{
				SepalLength = 3.3f,
				SepalWidth = 1.6f,
				PetalLength = 0.2f,
				PetalWidth = 5.1f,
			});

			Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
			Console.ReadLine();
		}
	}
}