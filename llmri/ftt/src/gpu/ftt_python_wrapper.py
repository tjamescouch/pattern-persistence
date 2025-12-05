import ctypes as ct
import os
import sys
import numpy as np
from pathlib import Path

# --- 1. C-API Interface Mapping ---

# Define the location of the shared library (.dylib or .so)
# Assumes the library is compiled into the 'build' directory adjacent to the script
def load_bit_pipeline():
    # Attempt to locate the shared library
    lib_path = Path.cwd() / "build" / "libftt_backend.dylib"
    if not lib_path.exists():
        # Fallback for common Linux/other OS extension
        lib_path = Path.cwd() / "build" / "libftt_backend.so"
    
    if not lib_path.exists():
        raise FileNotFoundError(f"ftt_backend library not found at: {lib_path.resolve()}")

    try:
        # Load the compiled shared library
        lib = ct.CDLL(str(lib_path))
    except Exception as e:
        print(f"Error loading C-library: {e}")
        sys.exit(1)

    # Map C-functions to Python
    # 1. init_bit_pipeline() -> returns opaque void* pointer (the BitPipeline instance)
    lib.init_bit_pipeline.restype = ct.c_void_p
    
    # 2. cleanup_bit_pipeline(pipe_ptr)
    lib.cleanup_bit_pipeline.argtypes = [ct.c_void_p]

    # 3. quantize_batch(pipe_ptr, src, dst, scales, rows, dim)
    # Uses numpy.ctypeslib.ndpointer for safe array passing
    _ndf = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    _ndi = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS", write=True)
    
    lib.quantize_batch.argtypes = [
        ct.c_void_p,  # pipe_ptr
        _ndf,         # src (input float32)
        _ndi,         # dst (output int8)
        _ndf,         # scales (output float32)
        ct.c_uint32,  # rows
        ct.c_uint32   # dim
    ]
    
    return lib

# --- 2. FTT_Writer (Integration Example) ---

class FTT_Writer_Metal:
    """
    Python side of the FTT writer, leveraging the Metal C-API.
    """
    def __init__(self, dim: int, max_rows: int = 10000):
        # Load the C-library and initialize the C++ BitPipeline object
        self.lib = load_bit_pipeline()
        self.pipe_ptr = self.lib.init_bit_pipeline()
        if not self.pipe_ptr:
            raise RuntimeError("Failed to initialize Metal BitPipeline.")
        
        self.dim = dim
        self.rows_in_batch = 0 # Example tracking
        print(f"[Python] BitPipeline loaded successfully. Dim: {dim}")

    def push_accelerated(self, tensor_f32: np.ndarray):
        """
        Accelerate quantization by calling the C++/Metal backend.
        
        Args:
            tensor_f32: numpy array [Rows, Dim] of float32 activations.
        """
        rows, dim = tensor_f32.shape
        if dim != self.dim:
            raise ValueError(f"Dimension mismatch: Expected {self.dim}, got {dim}")

        # Allocate output buffers in Python/Host memory
        # These are passed to C and filled by the Metal kernel
        dst_int8 = np.empty((rows, dim), dtype=np.int8)
        scales = np.empty(rows, dtype=np.float32)

        # Call the C function. The C-API handles the GPU dispatch.
        self.lib.quantize_batch(
            self.pipe_ptr,
            tensor_f32,
            dst_int8,
            scales,
            ct.c_uint32(rows),
            ct.c_uint32(dim)
        )
        
        # Now, dst_int8 and scales contain the GPU results
        # You would typically write them to disk here (using torch.save or raw bytes)
        print(f"[Python] Quantization complete for {rows} rows. Max scale: {scales.max():.4f}")
        
        return dst_int8, scales

    def __del__(self):
        # Ensure the C++ object is properly deleted when the Python object is destroyed
        if self.pipe_ptr:
            self.lib.cleanup_bit_pipeline(self.pipe_ptr)
            print("[Python] BitPipeline cleaned up.")


if __name__ == "__main__":
    print("--- FTT Python Wrapper Test ---")
    
    # 1. Simulate data capture (e.g., from an LLM forward pass)
    TEST_DIM = 4096
    TEST_ROWS = 32
    
    # Create test data (must be float32 for the Metal kernel)
    # Use high variance to ensure max/min is hit
    sim_input = np.random.randn(TEST_ROWS, TEST_DIM).astype(np.float32) * 8.0 
    
    try:
        # 2. Initialize and run
        writer = FTT_Writer_Metal(dim=TEST_DIM)
        output_int8, output_scales = writer.push_accelerated(sim_input)

        # 3. Validation Check
        # Check if the output contains clipped values (good sign for symmetric quantization)
        if 127 in output_int8 or -127 in output_int8:
            print("✅ TEST PASSED: Output contains clipped values (127/-127), confirming scale utilization.")
        else:
            print("❌ TEST WARNING: Output did not contain clipped values. Check input data variance.")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}\nPlease ensure 'libftt_backend.dylib' has been successfully compiled using CMake and Make.")
    except RuntimeError as e:
        print(f"\nFATAL ERROR: {e}")
