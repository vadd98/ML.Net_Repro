using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace MLNet_Repro.Models;

public class ModelInput
{
    [ImageType(256, 256)]
    public Bitmap Image { get; set; }
    
    public UInt32 LabelAsKey { get; set; } 

    public string Label { get; set; }
    
    public string ImagePath { get; set; }
}