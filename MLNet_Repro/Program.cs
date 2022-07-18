using System.Drawing;
using MLNet_Repro.Services;

Console.WriteLine("Starting");

string baseDirectory =
    Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? 
    AppDomain.CurrentDomain.BaseDirectory;


using var service = new MLNetPredictionService(baseDirectory + Path.DirectorySeparatorChar + "TrainedModel");
var imagePath = baseDirectory + Path.DirectorySeparatorChar + "Images" + Path.DirectorySeparatorChar + "example.png";
var image = Image.FromFile(imagePath) as Bitmap;

var prediction = service.Predict(image);

Console.WriteLine("Ended");