#!/usr/bin/env python3
"""
MRI Server - Streams neural activations from Anima over socket.

Usage in Anima:
    from mri_server import MRIServer
    server = MRIServer(port=9999)
    server.start()
    # During generation:
    server.broadcast(layer_idx, activations)

Usage standalone:
    python mri_server.py --port 9999
"""

import socket
import threading
import json
import struct
import time
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MRIScan:
    """Single layer activation snapshot."""
    timestamp: float
    layer_idx: int
    turn: int
    token_idx: int
    activations: np.ndarray  # Raw activation vector
    stats: Dict[str, float]  # min, max, mean, std


class MRIServer:
    """
    Broadcasts neural activations to connected clients.
    Also receives commands (interventions) from clients.
    
    Protocol:
    - Header: 4 bytes (uint32) = message length
    - Body: JSON with base64-encoded activations (outgoing) or commands (incoming)
    """
    
    def __init__(self, port: int = 9999, host: str = "localhost"):
        self.port = port
        self.host = host
        self.server_socket: Optional[socket.socket] = None
        self.clients: List[socket.socket] = []
        self.running = False
        self.turn = 0
        self.token_idx = 0
        
        # Hooks for integration
        self.on_client_connect: Optional[Callable] = None
        self.on_client_disconnect: Optional[Callable] = None
        self.on_command: Optional[Callable[[str, Dict], None]] = None  # For receiving commands
        
        # Buffer recent scans for new clients
        self.scan_buffer: List[MRIScan] = []
        self.buffer_size = 100
        
        self._lock = threading.Lock()
        self._client_threads: Dict[socket.socket, threading.Thread] = {}
    
    def start(self):
        """Start the server in background thread."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic checks
        
        self.running = True
        self._accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
        self._accept_thread.start()
        
        print(f"[MRI Server] Listening on {self.host}:{self.port}")
    
    def stop(self):
        """Stop the server."""
        self.running = False
        
        with self._lock:
            for client in self.clients:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        if self.server_socket:
            self.server_socket.close()
        
        print("[MRI Server] Stopped")
    
    def _accept_clients(self):
        """Accept new client connections."""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                with self._lock:
                    self.clients.append(client)
                print(f"[MRI Server] Client connected: {addr}")
                
                # Start receive thread for this client
                recv_thread = threading.Thread(
                    target=self._client_receive_loop, 
                    args=(client, addr),
                    daemon=True
                )
                self._client_threads[client] = recv_thread
                recv_thread.start()
                
                # Send recent buffer to new client
                self._send_buffer(client)
                
                if self.on_client_connect:
                    self.on_client_connect(addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[MRI Server] Accept error: {e}")
    
    def _client_receive_loop(self, client: socket.socket, addr):
        """Receive commands from a client."""
        client.settimeout(1.0)  # Allow periodic checks
        
        while self.running:
            try:
                # Read header
                header = self._recv_exactly(client, 4)
                if not header:
                    break
                
                msg_len = struct.unpack('!I', header)[0]
                if msg_len > 1024 * 1024:  # Sanity check: 1MB max
                    print(f"[MRI Server] Message too large from {addr}: {msg_len}")
                    break
                
                body = self._recv_exactly(client, msg_len)
                if not body:
                    break
                
                try:
                    data = json.loads(body.decode('utf-8'))
                    self._handle_client_command(data)
                except json.JSONDecodeError:
                    print(f"[MRI Server] Invalid JSON from {addr}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    # Connection closed or error
                    break
        
        # Client disconnected
        self._remove_client(client)
    
    def _recv_exactly(self, client: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from client."""
        data = b''
        while len(data) < n:
            try:
                chunk = client.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
            except:
                return None
        return data
    
    def _handle_client_command(self, data: Dict):
        """Handle command from client."""
        cmd_type = data.get("type", "")
        
        if cmd_type == "intervention":
            # Intervention command: set neuron multipliers
            interventions = data.get("interventions", {})
            layer = data.get("layer")
            
            print(f"[MRI Server] Received intervention: {len(interventions)} neurons, layer={layer}")
            
            if self.on_command:
                self.on_command("intervention", {
                    "interventions": interventions,
                    "layer": layer
                })
        
        elif cmd_type == "clear_interventions":
            print("[MRI Server] Received clear interventions")
            if self.on_command:
                self.on_command("clear_interventions", {})
        
        elif cmd_type == "ping":
            # Keepalive
            pass
        
        else:
            print(f"[MRI Server] Unknown command type: {cmd_type}")
    
    def _remove_client(self, client: socket.socket):
        """Remove a disconnected client."""
        with self._lock:
            if client in self.clients:
                self.clients.remove(client)
                print("[MRI Server] Client disconnected")
                if self.on_client_disconnect:
                    self.on_client_disconnect()
            if client in self._client_threads:
                del self._client_threads[client]
        
        try:
            client.close()
        except:
            pass
    
    def _send_buffer(self, client: socket.socket):
        """Send buffered scans to new client."""
        for scan in self.scan_buffer:
            self._send_to_client(client, scan)
    
    def _send_to_client(self, client: socket.socket, scan: MRIScan) -> bool:
        """Send scan to single client. Returns False if failed."""
        try:
            # Serialize
            data = {
                "type": "scan",
                "timestamp": scan.timestamp,
                "layer": scan.layer_idx,
                "turn": scan.turn,
                "token": scan.token_idx,
                "shape": list(scan.activations.shape),
                "dtype": str(scan.activations.dtype),
                "activations": scan.activations.tobytes().hex(),
                "stats": scan.stats
            }
            
            body = json.dumps(data).encode('utf-8')
            header = struct.pack('!I', len(body))
            
            client.sendall(header + body)
            return True
            
        except Exception as e:
            return False
    
    def broadcast(self, layer_idx: int, activations: np.ndarray):
        """Broadcast activation snapshot to all clients."""
        scan = MRIScan(
            timestamp=time.time(),
            layer_idx=layer_idx,
            turn=self.turn,
            token_idx=self.token_idx,
            activations=activations.astype(np.float32),
            stats={
                "min": float(activations.min()),
                "max": float(activations.max()),
                "mean": float(activations.mean()),
                "std": float(activations.std()),
            }
        )
        
        # Add to buffer
        self.scan_buffer.append(scan)
        if len(self.scan_buffer) > self.buffer_size:
            self.scan_buffer.pop(0)
        
        # Broadcast to clients
        dead_clients = []
        
        with self._lock:
            for client in self.clients:
                if not self._send_to_client(client, scan):
                    dead_clients.append(client)
            
            # Remove dead clients
            for client in dead_clients:
                self.clients.remove(client)
                print(f"[MRI Server] Client disconnected")
                if self.on_client_disconnect:
                    self.on_client_disconnect()
    
    def broadcast_event(self, event_type: str, data: Dict):
        """Broadcast non-scan event (turn start, generation complete, etc)."""
        msg = {
            "type": event_type,
            "timestamp": time.time(),
            **data
        }
        
        body = json.dumps(msg).encode('utf-8')
        header = struct.pack('!I', len(body))
        
        dead_clients = []
        
        with self._lock:
            for client in self.clients:
                try:
                    client.sendall(header + body)
                except:
                    dead_clients.append(client)
            
            for client in dead_clients:
                self.clients.remove(client)
    
    def new_turn(self):
        """Signal start of new turn."""
        self.turn += 1
        self.token_idx = 0
        self.broadcast_event("turn_start", {"turn": self.turn})
    
    def next_token(self):
        """Increment token counter."""
        self.token_idx += 1
    
    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self.clients)


# Standalone test server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--test", action="store_true", help="Send test data")
    args = parser.parse_args()
    
    server = MRIServer(port=args.port)
    server.start()
    
    if args.test:
        print("[Test mode] Sending synthetic activations...")
        try:
            while True:
                for layer in range(46):
                    # Synthetic activation pattern
                    t = time.time()
                    activations = np.sin(np.arange(4608) * 0.01 + t + layer * 0.1).astype(np.float32)
                    server.broadcast(layer, activations)
                    time.sleep(0.05)
                
                server.next_token()
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
    else:
        print("Server running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    server.stop()