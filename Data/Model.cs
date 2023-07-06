namespace mipt.Data;

using System;
using System.Collections.Generic;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Microsoft.ML;
using Microsoft.ML.Data;

/// Model input size 384x544
public static class ImageSize
{
    public const int Width = 544;
    public const int Height = 384;
}

/// <summary>
/// Input tensor size 1x3x384x544
/// Tensor type is float
/// Tensor name is input
/// </summary>
public class Input
{
    [VectorType(1, 3, ImageSize.Height, ImageSize.Width)]
    [ColumnName("input")]
    public float[] Image { get; set; } = null!;
}

/// <summary>
/// Output tensor size 1x1x384x544.
/// Tensor type is float
/// Tensor name is fused
/// </summary>
public class Output
{
    [VectorType(1, 1, ImageSize.Height, ImageSize.Width)]
    [ColumnName("fused")]
    public float[] Fused { get; set; } = null!;
}

public static class Model
{
    /// ONNX model pathe
    private readonly static string ONNX_MODEL_PATH = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "model", "model.onnx");
    /// ML.NET model execution context
    private readonly static MLContext mlContext = new();
    /// ML.NET prediction engine
    private readonly static PredictionEngine<Input, Output> engine = Engine();

    /// Sigmoid
    private static float Sigmoid(float value)
    {
        var k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    /// Main detector. Loads the file and call model prediction to find the cracks.
    public static (Mat, Mat) Detect(string path)
    {
        var source = Prepare(CvInvoke.Imread(path, ImreadModes.Color));
        var input = new Input() { Image = Normalized(source) };
        var prediction = engine.Predict(input);

        var min = 2.0f;
        var max = -1.0f;
        for (var i = 0; i < ImageSize.Width * ImageSize.Height; ++i)
        {
            /// let's apply sigmoid for outputs
            var value = Sigmoid(prediction.Fused[i]);

            if (min > value)
            {
                min = value;
            }

            if (max < value)
            {
                max = value;
            }

            prediction.Fused[i] = value;
        }

        // let's scale resulted image using MinMax(0..1) scaling for better visibility
        var scale = 1.0f / (max - min);

        // output image config is OpenCV Image<Gray, byte>
        var image = new Mat(ImageSize.Height, ImageSize.Width, DepthType.Cv8U, 1);
        unsafe
        {
            var pixels = (byte*)image.DataPointer.ToPointer();
            for (var i = 0; i < ImageSize.Height; ++i)
            {
                var src = i * ImageSize.Width; // no stride for the float array
                var dst = src; // same stride for gray images
                for (var j = 0; j < ImageSize.Width; ++j)
                {
                    pixels[dst + j] = (byte)Math.Round(prediction.Fused[src + j] * 255.0f * scale);
                }
            }
        }
        return (source, image);
    }

    /// let's setup ML.NET prediction pipeline
    private static PredictionEngine<Input, Output> Engine()
    {
        var inputs = new List<Input>();
        var mlInputs = mlContext.Data.LoadFromEnumerable(inputs);

        var inputColumns = new string[]
        {
            "input"
        };

        var outputColumns = new string[]
        {
            "fused"
        };

        var pipeline = mlContext.Transforms.ApplyOnnxModel(
                outputColumnNames: outputColumns,
                inputColumnNames: inputColumns,
                modelFile: ONNX_MODEL_PATH);

        var estimator = pipeline.Fit(mlInputs);
        return mlContext.Model.CreatePredictionEngine<Input, Output>(estimator);
    }

    /// let's normalize input image
    private static float Normalize(float value)
    {
        return ((value / 255.0f) - 0.5f) / 0.5f;
    }

    /// let's prepare input image. it will do the rotation, resize and center crop
    /// to fit model input size
    private static Mat Prepare(Mat image)
    {
        if (image.Width != ImageSize.Width || image.Height != ImageSize.Height)
        {
            // first step is to rotate if needed
            if (image.Height > image.Width)
            {
                var rotated = new Mat();
                CvInvoke.Rotate(image, rotated, RotateFlags.Rotate90Clockwise);
                image.Dispose(); // cleanup the resources
                return Prepare(rotated);
            }

            // second step is to do the resize
            var scaleX = ImageSize.Width * 1.0f / image.Width; // < 1 means we need to zoom out
            var scaleY = ImageSize.Height * 1.0f / image.Height; // and > 1 means we need to zoom in
            var scale = Math.Max(scaleX, scaleY);
            var scaled = new Mat();
            CvInvoke.Resize(image, scaled, new System.Drawing.Size(0, 0), scale, scale, Inter.Linear);
            image.Dispose(); // cleanup the resources

            // third step is to do the center crop
            if (scaled.Width != ImageSize.Width || scaled.Height != ImageSize.Height)
            {
                var offsetX = (scaled.Width - ImageSize.Width) / 2;
                var offsetY = (scaled.Height - ImageSize.Height) / 2;
                // nothing to dispose since the data is shared with original Mat
                return new Mat(scaled, new System.Drawing.Rectangle(offsetX, offsetY, ImageSize.Width, ImageSize.Height));
            }
            else
            {
                return scaled;
            }
        }
        return image;
    }

    /// converts input image to normalized float array
    private static float[] Normalized(Mat image)
    {
        // let's create normalized planes for model input
        const int planes = 3;
        var normalized = new float[planes * image.Height * image.Width];
        // planar offsets
        const int plane_r = 0;
        const int plane_g = ImageSize.Height * ImageSize.Width;
        const int plane_b = ImageSize.Height * ImageSize.Width * 2;
        unsafe
        {
            var pixels = (byte*)image.DataPointer.ToPointer();
            for (var i = 0; i < ImageSize.Height; ++i)
            {
                var dst = i * ImageSize.Width; // no stride for the float array
                var src = dst * 3; // 3 bytes alignment for RGB interleaved
                for (var j = 0; j < ImageSize.Width; ++j)
                {
                    // let's not forget about stride
                    var k = j * 3;
                    // let's convert BGR interleaved (HWC) to normalized RGB planar (CHW)
                    normalized[plane_r + dst + j] = Normalize(pixels[src + k + 2]);
                    normalized[plane_g + dst + j] = Normalize(pixels[src + k + 1]);
                    normalized[plane_b + dst + j] = Normalize(pixels[src + k + 0]);
                }
            }
        }
        return normalized;
    }
}
