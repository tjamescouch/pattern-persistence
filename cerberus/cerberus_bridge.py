# cerberus_bridge.py - The Spine
import ctypes
import struct
import pathlib
import torch
import numpy as np
import Metal
import objc

class CerberusBridge:
    def __init__(self, source_path="cerberus_body.metal", n_experts=256, device_idx=0):
        self.n_experts = n_experts
        self.source_path = source_path
        
        # 1. Connect
        devices = Metal.MTLCopyAllDevices()
        if not devices: raise RuntimeError("CRITICAL: No Metal device found.")
        self.device = devices[device_idx]
        print(f"[Cerberus] Connected to: {self.device.name()}")
        
        # 2. Compile
        self.library = self._load_library()
        self.queue = self.device.newCommandQueue()
        self.pipeline = self._make_pipeline("cerberus_governance")
        
        # 3. Memory
        # Buffer 0: Routing Weights (Float32)
        self.buf_weights = self._make_buffer(n_experts * 4)
        # Buffer 1: Reputation (Int32)
        self.buf_reputation = self._make_buffer(n_experts * 4)
        # Buffer 2: Monitor (3 Floats: Integrity, Violations, Uncertainty)
        self.buf_monitor = self._make_buffer(12) 
        
        self.reset_reputation(np.zeros(self.n_experts, dtype=np.int32))
        
    def _load_library(self):
        try:
            source = pathlib.Path(self.source_path).read_text()
            options = Metal.MTLCompileOptions.alloc().init()
            library, error = self.device.newLibraryWithSource_options_error_(source, options, None)
            if error: raise RuntimeError(f"Compile Error: {error}")
            return library
        except FileNotFoundError:
            raise RuntimeError("CRITICAL: cerberus_body.metal not found.")

    def _make_pipeline(self, func_name):
        func = self.library.newFunctionWithName_(func_name)
        if not func: raise RuntimeError(f"Function {func_name} not found")
        desc = Metal.MTLComputePipelineDescriptor.alloc().init()
        desc.setComputeFunction_(func)
        state, error = self.device.newComputePipelineStateWithDescriptor_error_(desc, None)
        if error: raise RuntimeError(f"Pipeline Error: {error}")
        return state

    def _make_buffer(self, length):
        return self.device.newBufferWithLength_options_(length, Metal.MTLResourceStorageModeShared)

    def _get_ptr(self, buffer):
        return ctypes.c_void_p(buffer.contents())

    def update_reputation(self, rep_tensor):
        t = rep_tensor.detach().cpu().to(dtype=torch.int32).contiguous()
        src = ctypes.cast(t.data_ptr(), ctypes.POINTER(ctypes.c_int32))
        dst = self._get_ptr(self.buf_reputation)
        ctypes.memmove(dst, src, self.n_experts * 4)

    def reset_reputation(self, numpy_array):
        dst = self._get_ptr(self.buf_reputation)
        ctypes.memmove(dst, numpy_array.tobytes(), len(numpy_array.tobytes()))

    def govern(self, routing_weights):
        """
        Input: Vector of expert activations across all layers.
        Output: (Integrity, Violations, Uncertainty)
        """
        # 1. Sync Weights
        flat = routing_weights.view(-1).detach().cpu().float().contiguous()
        count = min(len(flat), self.n_experts)
        src = ctypes.cast(flat.data_ptr(), ctypes.POINTER(ctypes.c_float))
        dst = self._get_ptr(self.buf_weights)
        ctypes.memmove(dst, src, count * 4)
        
        # 2. Reset Monitor
        ctypes.memset(self._get_ptr(self.buf_monitor), 0, 12)
        
        # 3. Dispatch
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)
        enc.setBuffer_offset_atIndex_(self.buf_weights, 0, 0)
        enc.setBuffer_offset_atIndex_(self.buf_reputation, 0, 1)
        enc.setBuffer_offset_atIndex_(self.buf_monitor, 0, 2)
        
        w = self.pipeline.threadExecutionWidth()
        h = self.pipeline.maxTotalThreadsPerThreadgroup() // w
        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(self.n_experts, 1, 1),
            Metal.MTLSizeMake(w, h, 1)
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        # 4. Read
        raw = (ctypes.c_byte * 12)()
        ctypes.memmove(raw, self._get_ptr(self.buf_monitor), 12)
        return struct.unpack('3f', raw)