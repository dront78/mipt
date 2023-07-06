namespace mipt.Data;

public class DetectService
{
// #if DEBUG
//     // dlopen hack to debug on Linux
//     private const int RTLD_NOW = 2; // for dlopen's flags
//     private const int RTLD_GLOBAL = 8;
// 
//     [System.Runtime.InteropServices.DllImport("libdl.so.2")]
//     private static extern IntPtr dlopen(string filename, int flags);
// 
//     static DetectService()
//     {
//         dlopen(Path.Combine(Directory.GetCurrentDirectory(), "runtimes", "ubuntu-x64", "native", "libcvextern.so"), RTLD_NOW | RTLD_GLOBAL);
//     }
// #endif

    /// async task wrapper to avoid freezing web ui.
    public Task<string> DetectAsync(string filename)
    {
        return Task.FromResult(Detect(filename));
    }

    /// detect pipeline. returns processed filename.
    public static string Detect(string filename)
    {
        var (source, image) = Model.Detect(filename);
        Processor.DrawContours(source, image);
        string result = filename + ".processed" + Path.GetExtension(filename);
        source.Save(result);
        return result;
    }
}
