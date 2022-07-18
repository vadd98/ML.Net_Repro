using Microsoft.ML.Data;

namespace MLNet_Repro.Models.Training;

public class TensorflowInput
{
    public const string INPUT_NODE_NAME = "serving_default_conv2d_input";
    public const string SAVER_FILENAME_NODE = "saver_filename";
    
    [ColumnName(INPUT_NODE_NAME), VectorType(256, 256, 1)]
    public float[] Input { get; set; }

    [ColumnName(SAVER_FILENAME_NODE)] 
    public string SaverFileName { get; set; }
}