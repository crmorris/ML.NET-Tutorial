using System;
using Microsoft.ML;

//We need an instance of the context to use ML.NET
var context = new MLContext();

//Loads in our data
var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

//We are splitting our data here, so we can teach the model with one section and test the model on another section
var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

//"These are the columns we give our algorithm to allow it to use the data"
//Label is filtered out here because it's our target column; OceanProximity is filtered out because it's a string
//and every other column is a float
var features = split.TrainSet.Schema
    .Select(col => col.Name)
    .Where(colName => colName != "Label" && colName != "OceanProximity")
    .ToArray();

//Pre-processing data and choosing ML algorithm
//"FeaturizeText" normalizes the string values
var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
    //For the pipeline, just append as many things as necessary
    //This line concatenates the feature query above and adds it to the "Features" column
    .Append(context.Transforms.Concatenate("Features", features))
    .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
    //This selects the regression method
    .Append(context.Regression.Trainers.LbfgsPoissonRegression());

//This trains the model
var model = pipeline.Fit(split.TrainSet);

//Uses the newly trained model to make predictions
var predictions = model.Transform(split.TestSet);

//Evaluates the accuracy of the predictions
var metrics = context.Regression.Evaluate(predictions);

Console.WriteLine($"R^2 - {metrics.RSquared}");

//Summary - creates context, loads data, splits data, adds column, pre-processes/chooses algorithm,
//trains the model, tests against predictions, prints results

//Also important to know - add model builder extension, change CSV property to "copy if newer"