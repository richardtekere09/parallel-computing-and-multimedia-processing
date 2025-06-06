import pyopencl as cl
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# OpenCL Kernel: Emboss + Downscale (2x2)
kernel_code = """
__kernel void emboss_downsample_1d(__global const float* in_img, __global float* out_img, int W, int H) {
    int gid = get_global_id(0);
    int outW = W/2, outH = H/2;
    int c = gid / (outW*outH);
    int rem = gid % (outW*outH);
    int out_y = rem / outW;
    int out_x = rem % outW;
    if (c>=3 || out_x>=outW || out_y>=outH) return;
    const float k[3][3] = {{-2,-1,0},{-1,1,1},{0,1,2}};
    int ix=2*out_x, iy=2*out_y;
    float sum=0;
    for(int dy=-1; dy<=1; dy++)
      for(int dx=-1; dx<=1; dx++) {
        int sx=clamp(ix+dx,0,W-1);
        int sy=clamp(iy+dy,0,H-1);
        sum += in_img[(sy*W+sx)*3+c]*k[dy+1][dx+1];
      }
    out_img[(out_y*outW+out_x)*3+c]=sum;
}
"""


def run_opencl(input_path, output_path, ctx, queue, prg):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width // 2, height // 2))

    mf = cl.mem_flags
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame.astype(np.float32).ravel()
        in_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
        out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes // 4)
        global_size = (width // 2 * height // 2 * 3,)
        prg.emboss_downsample_1d(
            queue, global_size, None, in_buf, out_buf, np.int32(width), np.int32(height)
        )
        result = np.empty((height // 2, width // 2, 3), dtype=np.float32)
        cl.enqueue_copy(queue, result.ravel(), out_buf)
        out_writer.write(result.clip(0, 255).astype(np.uint8))
    cap.release()
    out_writer.release()
    return (time.time() - start) * 1000  # ms


def main():
    input_path = "data/lab2.mov" 
    output_dir = os.path.join("output", "lab2")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Input video not found: {input_path}")
        return

    platforms = cl.get_platforms()
    for p in platforms:
        print(f"Platform: {p.name}")
        devices = p.get_devices()
        for d in devices:
            print(f"Device: {d.name} (Type: {cl.device_type.to_string(d.type)})")

    # OpenCL setup: GPU-only
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found. Exiting.")
        return
    devices = platforms[0].get_devices()
    gpu_devices = [d for d in devices if d.type & cl.device_type.GPU]
    if not gpu_devices:
        print("No GPU devices found! This lab requires a GPU. Exiting.")
        return
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, kernel_code).build()
    print(f"Using GPU: {gpu_devices[0].name}")


    times = []
    print("Running OpenCL Emboss + Downscale benchmark (GPU-only)...")
    for i in range(3):
        output_path = os.path.join(output_dir, f"output_run{i + 1}.mp4")
        t = run_opencl(input_path, output_path, ctx, queue, prg)
        times.append(t)
        print(f"Run {i + 1}: {t:.2f} ms")

    avg_time = sum(times) / 3
    print(f"Average Time: {avg_time:.2f} ms")

    # Save execution time visualization
    plt.figure(figsize=(8, 6))
    x_labels = ["Run 1", "Run 2", "Run 3", "Average"]
    y_values = times + [avg_time]
    plt.bar(x_labels, y_values, color="salmon")
    plt.ylabel("Execution Time (ms)")
    plt.title("OpenCL Emboss + Downscale Benchmark (GPU-only)")
    plt.grid(axis="y")
    plt_path = os.path.join(output_dir, "execution_times.png")
    plt.savefig(plt_path)
    plt.show()
    print(f"Visualization saved at: {plt_path}")


if __name__ == "__main__":
    main()
