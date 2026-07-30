"""
orchestrator
------------
The Adaptive Orchestrator for InferenceOS.
Loads the hardware profile, selects the optimal model quantization,
calculates VRAM layer offloading, and invokes the C++ backend via ctypes.
"""

__version__ = "0.1.0"
