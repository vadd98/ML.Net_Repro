using Microsoft.ML.Data;

namespace MLNet_Repro.Models.Training;

public class TensorflowOutput
{
    public const string OUTPUT_NODE_NAME = "StatefulPartitionedCall";

    [ColumnName(OUTPUT_NODE_NAME), VectorType(2)]
    public float[] Output { get; set; }
}