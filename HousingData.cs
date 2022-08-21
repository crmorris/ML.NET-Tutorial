using Microsoft.ML.Data;

//This is where you define the schema for your data
internal class HousingData
{
    //LoadColumn loads the respective column from the dataset
    [LoadColumn(0)]
    public float Longitude { get; set; }

    [LoadColumn(1)]
    public float Latitude { get; set; }

    [LoadColumn(2)]
    public float HousingMedianAge { get; set; }

    [LoadColumn(3)]
    public float TotalRooms { get; set; }

    [LoadColumn(4)]
    public float TotalBedrooms { get; set; }

    [LoadColumn(5)]
    public float Population { get; set; }

    [LoadColumn(6)]
    public float Households { get; set; }

    [LoadColumn(7)]
    public float MedianIncome { get; set; }

    //Because this is the data we want to predict, we also give it the attribute of "ColumnName"
    [LoadColumn(8), ColumnName("Label")]
    public float MedianHouseValue { get; set; }

    [LoadColumn(9)]
    public string OceanProximity { get; set; }
}