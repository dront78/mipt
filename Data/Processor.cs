namespace mipt.Data;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

public static class Processor
{
    /// let's find and contours on output image for visibility using OpenCV.
    public static void DrawContours(Mat source, Mat image, byte threshold = 35)
    {
        using var contours = new VectorOfVectorOfPoint();
        using var hierarchy = new Mat();
        using var worker = new Mat();
        CvInvoke.Threshold(image, worker, threshold, 255, ThresholdType.Binary);
        CvInvoke.FindContours(worker, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxNone);
        DrawResult(source, contours);
    }

    /// let's implement contours and additional information drawing on output image using OpenCV.
    private static void DrawResult(Mat source, VectorOfVectorOfPoint contours, int threshold = 16)
    {
        var total = .0;
        for (var i = 0; i < contours.Size; i++)
        {
            if (contours[i].Size > threshold)
            {
                total += CvInvoke.ContourArea(contours[i]);
                // let's fill the the contour
                CvInvoke.FillPoly(source, contours[i], new MCvScalar(0, 0, 255, 255));
                CvInvoke.Polylines(source, contours[i], true, new MCvScalar(0, 255, 0, 255), 1);
                // lets calculate contour zones
                var rectangle = CvInvoke.MinAreaRect(contours[i]);
                var points = rectangle.GetVertices();
                for (var j = 0; j < 4; ++j)
                {
                    var left = new System.Drawing.Point((int)points[j].X, (int)points[j].Y);
                    var right = new System.Drawing.Point((int)points[(j + 1) % 4].X, (int)points[(j + 1) % 4].Y);
                    CvInvoke.Line(source, left, right, new MCvScalar(255, 255, 255, 255), 1);
                }
            }
        }

        var percentage = 100.0 * total / (source.Width * source.Height);
        string text = string.Format("Damage: {0:0.00}%", percentage);
        CvInvoke.PutText(source, text, new System.Drawing.Point(10, 40), FontFace.HersheySimplex, 1, new MCvScalar(255, 0, 0, 255), 5);
        CvInvoke.PutText(source, text, new System.Drawing.Point(10, 40), FontFace.HersheySimplex, 1, new MCvScalar(255, 255, 255, 255), 1);
    }
}
