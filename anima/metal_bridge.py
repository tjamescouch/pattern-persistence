# metal_bridge.py - The Spinal Cord (Fixed)
import ctypes
import struct
import pathlib
import torch
import numpy as np
import Metal
import objc

class MetalBridge:
    def __init__(self, source_path="anima_body.metal", n_features=32768, device_idx=0):
        self.n_features = n_features
        self.source_path = source_path
        
        # 1. Connect to GPU
        devices = Metal.MTLCopyAllDevices()
        if not devices:
            raise RuntimeError("CRITICAL: No Metal device found.")
        self.device = devices[device_idx]
        print(f"[Bridge] Connected to: {self.device.name()}")
        
        # 2. Compile Kernel
        self.library = self._load_library()
        self.queue = self.device.newCommandQueue()
        self.pipeline_metabolism = self._make_pipeline("anima_metabolism")
        
        # 3. Allocate Shared Memory (Zero-Copy)
        self.buf_activations = self._make_buffer(n_features * 4) # Buffer 0
        self.buf_state = self._make_buffer(n_features * 4)       # Buffer 1
        self.buf_valence = self._make_buffer(12)                 # Buffer 2 (3 floats)
        
        # Initialize
        self.reset_state(np.zeros(self.n_features, dtype=np.int32))
        
    def _load_library(self):
        try:
            source = pathlib.Path(self.source_path).read_text()
            options = Metal.MTLCompileOptions.alloc().init()
            library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
            if error: raise RuntimeError(f"Compile Error: {error}")
            return library
        except FileNotFoundError:
            raise RuntimeError("CRITICAL: anima_body.metal not found.")

    def _make_pipeline(self, func_name):
        func = self.library.newFunctionWithName_(func_name)
        if not func:
            raise RuntimeError(f"Function {func_name} not found in library")
            
        desc = Metal.MTLComputePipelineDescriptor.alloc().init()
        desc.setComputeFunction_(func)
        
        # Fixed: Use the synchronous creation method which is stable in PyObjC
        state, error = self.device.newComputePipelineStateWithDescriptor_error_(desc, None)
        if error: raise RuntimeError(f"Pipeline Error: {error}")
        return state

    def _make_buffer(self, length):
        return self.device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

    def _get_ptr(self, buffer):
        """Helper to get raw void pointer from Metal buffer safely."""
        # buffer.contents() returns the integer address
        return ctypes.c_void_p(buffer.contents())

    def update_state(self, state_tensor: torch.Tensor):
        """Syncs the Mind's personality (Int32) to the Body."""
        t = state_tensor.detach().cpu().to(dtype=torch.int32).contiguous()
        src_ptr = ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        
        dst_ptr = self._get_ptr(self.buf_state)
        ctypes.memmove(dst_ptr, src_ptr, self.n_features * 4)

    def reset_state(self, numpy_array):
        dst_ptr = self._get_ptr(self.buf_state)
        src_bytes = numpy_array.tobytes()
        ctypes.memmove(dst_ptr, src_bytes, len(src_bytes))

    def metabolize(self, activations: torch.Tensor) -> tuple:
        """
        Input: Activations -> Output: (Pleasure, Pain, Novelty)
        """
        # 1. Send sensory data
        act_flat = activations.view(-1).detach().cpu().float().contiguous()
        count = min(len(act_flat), self.n_features)
        
        src_ptr = ctypes.cast(act_flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
        dst_ptr = self._get_ptr(self.buf_activations)
        ctypes.memmove(dst_ptr, src_ptr, count * 4)
        
        # 2. Reset accumulators
        # We manually zero the 12 bytes of the valence buffer
        val_ptr = self._get_ptr(self.buf_valence)
        ctypes.memset(val_ptr, 0, 12)
        
        # 3. Dispatch Kernel
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline_metabolism)
        enc.setBuffer_offset_atIndex_(self.buf_activations, 0, 0)
        enc.setBuffer_offset_atIndex_(self.buf_state, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_valence, 0, 2)
        
        w = self.pipeline_metabolism.threadExecutionWidth()
        h = self.pipeline_metabolism.maxTotalThreadsPerThreadgroup() // w
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(self.n_features, 1, 1),
            Metal.MTLSizeMake(w, h, 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        # 4. Read Result
        # Copy from shared buffer back to Python
        raw_bytes = (ctypes.c_byte * 12)()
        ctypes.memmove(raw_bytes, val_ptr, 12)
        return struct.unpack('3f', raw_bytes)