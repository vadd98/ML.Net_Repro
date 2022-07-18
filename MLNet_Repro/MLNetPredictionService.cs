using System.Drawing;
using MLNet_Repro.Models;
using MLNet_Repro.Models.Training;
using Microsoft.ML;

namespace MLNet_Repro.Services;

public class MLNetPredictionService 
{
    private readonly MLContext _mlContext;
    private PredictionEngine<ModelInput, TensorflowOutput> _predictionEngine;

    public MLNetPredictionService(string modelPath)
    {
        _mlContext = new MLContext();
        LoadTensorflowModel(modelPath); 
    }
    
    #region TENSORFLOW
    
    private void LoadTensorflowModel(string tensorflowModelPath)
    {
        var tensorflowModel = _mlContext.Model.LoadTensorFlowModel(tensorflowModelPath);

        var scoreModel = tensorflowModel.ScoreTensorFlowModel(
            inputColumnNames: new []{TensorflowInput.INPUT_NODE_NAME, TensorflowInput.SAVER_FILENAME_NODE},
            outputColumnNames: new []{TensorflowOutput.OUTPUT_NODE_NAME},
            addBatchDimensionInput: false);

        // need to put some data in SAVER_FILENAME_NODE, otherwise will get exception
        var pipeline = _mlContext.Transforms
            .CopyColumns(
                inputColumnName: nameof(ModelInput.ImagePath),
                outputColumnName: TensorflowInput.SAVER_FILENAME_NODE)
            .Append(_mlContext.Transforms
                .ExtractPixels(
                    inputColumnName: nameof(ModelInput.Image),
                    outputColumnName: TensorflowInput.INPUT_NODE_NAME))
            .Append(scoreModel);

        IDataView dataView = _mlContext.Data
            .LoadFromEnumerable(new List<ModelInput>());
        
        ITransformer model = pipeline.Fit(dataView);
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, TensorflowOutput>(model);
    }

    public object Predict(Bitmap spectrogram)
    {
        ModelInput inputModel = new ModelInput()
        {
            ImagePath = "",
            Label = "",
            LabelAsKey = 12,
            Image = spectrogram
        };

        // TODO crash here
        var prediction = _predictionEngine.Predict(inputModel);
        var output = prediction.Output
            .Take(2)
            .ToList();
        
        Console.WriteLine("Obtained result");

        return output;
    }

    #endregion

    public void Dispose()
    {
        _predictionEngine?.Dispose();
    }
}