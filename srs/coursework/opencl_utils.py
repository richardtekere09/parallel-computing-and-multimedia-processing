try:
    import pyopencl as cl
    import numpy as np
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

class OpenCLAccelerator:
    """OpenCL-based neural network acceleration"""
    
    def __init__(self):
        if not OPENCL_AVAILABLE:
            raise ImportError("PyOpenCL not available")
        
        # Initialize OpenCL context
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)
        
        # Compile kernels
        self.program = cl.Program(self.context, self._get_kernel_code()).build()
        
    def _get_kernel_code(self) -> str:
        """OpenCL kernel for matrix operations"""
        return """
        __kernel void matrix_multiply(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int N, const int M, const int K) {
            
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row < N && col < M) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * M + col];
                }
                C[row * M + col] = sum;
            }
        }
        
        __kernel void apply_activation(
            __global float* data,
            const int size,
            const int activation_type) {
            
            int idx = get_global_id(0);
            if (idx < size) {
                float x = data[idx];
                
                if (activation_type == 0) {  // tanh
                    data[idx] = tanh(x);
                } else if (activation_type == 1) {  // relu
                    data[idx] = fmax(0.0f, x);
                } else if (activation_type == 2) {  // sigmoid
                    data[idx] = 1.0f / (1.0f + exp(-x));
                }
                // activation_type == 3: linear (no change)
            }
        }
        """
    
    def accelerated_forward_pass(self, inputs: np.ndarray, 
                               weights: np.ndarray, 
                               biases: np.ndarray,
                               activation_type: int) -> np.ndarray:
        """GPU-accelerated forward pass"""
        
        # Prepare data
        inputs_gpu = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=inputs)
        weights_gpu = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=weights)
        
        # Output buffer
        output_shape = (inputs.shape[0], weights.shape[1])
        output = np.zeros(output_shape, dtype=np.float32)
        output_gpu = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        # Matrix multiplication
        global_size = output_shape
        self.program.matrix_multiply(self.queue, global_size, None,
                                   inputs_gpu, weights_gpu, output_gpu,
                                   np.int32(output_shape[0]),
                                   np.int32(output_shape[1]),
                                   np.int32(inputs.shape[1]))
        
        # Apply activation
        self.program.apply_activation(self.queue, (output.size,), None,
                                    output_gpu, np.int32(output.size),
                                    np.int32(activation_type))
        
        # Read result
        cl.enqueue_copy(self.queue, output, output_gpu)
        
        return output