"""
Resource monitoring module for tracking memory and GPU usage.
"""
import os
import gc
import psutil
import torch
import logging
from typing import Optional, Tuple, Dict, Any

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        dict: Dictionary with memory usage statistics (percentages)
    """
    process = psutil.Process(os.getpid())
    ram_info = psutil.virtual_memory()
    
    stats = {
        "ram_percent": ram_info.percent,
        "ram_used_gb": ram_info.used / (1024 ** 3),
        "ram_total_gb": ram_info.total / (1024 ** 3),
        "process_ram_gb": process.memory_info().rss / (1024 ** 3),
        "process_ram_percent": (process.memory_info().rss / ram_info.total) * 100
    }
    
    return stats

def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage statistics if available.
    
    Returns:
        dict: Dictionary with GPU memory usage statistics, or empty if no GPU
    """
    if not torch.cuda.is_available():
        return {}
    
    try:
        # Get current device ID
        device_id = torch.cuda.current_device()
        
        # Get memory usage statistics
        gpu_stats = {
            "gpu_used_gb": torch.cuda.memory_allocated(device_id) / (1024 ** 3),
            "gpu_cached_gb": torch.cuda.memory_reserved(device_id) / (1024 ** 3),
            "gpu_total_gb": torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3),
            "gpu_percent": (torch.cuda.memory_allocated(device_id) / 
                           torch.cuda.get_device_properties(device_id).total_memory) * 100
        }
        
        return gpu_stats
    except Exception as e:
        return {"gpu_error": str(e)}

def log_memory_usage(logger: logging.Logger) -> None:
    """
    Log current memory usage statistics.
    
    Args:
        logger: Logger to use for logging
    """
    mem_stats = get_memory_usage()
    gpu_stats = get_gpu_memory_usage()
    
    # Log memory usage
    logger.info(f"RAM: {mem_stats['ram_percent']:.1f}% used, "
                f"Process: {mem_stats['process_ram_gb']:.2f} GB ({mem_stats['process_ram_percent']:.1f}%)")
    
    # Log GPU usage if available
    if gpu_stats and "gpu_error" not in gpu_stats:
        logger.info(f"GPU: {gpu_stats['gpu_percent']:.1f}% used, "
                    f"{gpu_stats['gpu_used_gb']:.2f} GB allocated, "
                    f"{gpu_stats['gpu_cached_gb']:.2f} GB cached")

def check_gpu_availability() -> Tuple[bool, torch.device]:
    """
    Check if GPU is available for computation.
    
    Returns:
        tuple: (is_gpu_available, device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        return True, device
    else:
        device = torch.device("cpu")
        return False, device

def free_memory(logger: Optional[logging.Logger] = None) -> None:
    """
    Free unused memory by forcing garbage collection and emptying PyTorch CUDA cache.
    
    Args:
        logger: Optional logger to use for logging
    """
    # Force Python garbage collection
    gc.collect()
    
    # Empty PyTorch CUDA cache if GPU is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger:
            logger.info("Freed GPU memory cache")
    
    if logger:
        logger.info("Forced garbage collection")

class ResourceManager:
    """
    Resource manager for monitoring and managing system resources.
    """
    
    def __init__(self, max_memory_percentage: float = 80.0, max_gpu_percentage: float = 80.0):
        """
        Initialize the resource manager.
        
        Args:
            max_memory_percentage: Maximum memory percentage before unloading models
            max_gpu_percentage: Maximum GPU memory percentage before unloading models
        """
        self.max_memory_percentage = max_memory_percentage
        self.max_gpu_percentage = max_gpu_percentage
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}
        
    def register_model(self, model_id: str, model: Any, model_size_gb: float, 
                      priority: int = 1, uses_gpu: bool = True) -> None:
        """
        Register a model with the resource manager.
        
        Args:
            model_id: Unique identifier for the model
            model: The model object
            model_size_gb: Approximate size of the model in GB
            priority: Priority (1-10, higher means more important to keep)
            uses_gpu: Whether the model uses GPU
        """
        self.loaded_models[model_id] = {
            "model": model,
            "size_gb": model_size_gb,
            "priority": priority,
            "uses_gpu": uses_gpu,
            "last_used": 0  # Will be updated when model is used
        }
        self.logger.info(f"Registered model {model_id} (size: {model_size_gb:.2f} GB, priority: {priority})")
        
    def unregister_model(self, model_id: str) -> None:
        """
        Unregister a model from the resource manager.
        
        Args:
            model_id: Unique identifier for the model
        """
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self.logger.info(f"Unregistered model {model_id}")
    
    def mark_model_used(self, model_id: str) -> None:
        """
        Mark a model as recently used.
        
        Args:
            model_id: Unique identifier for the model
        """
        if model_id in self.loaded_models:
            import time
            self.loaded_models[model_id]["last_used"] = time.time()
    
    def check_resource_limits(self) -> Tuple[bool, bool]:
        """
        Check if system is approaching resource limits.
        
        Returns:
            tuple: (memory_critical, gpu_critical)
        """
        mem_stats = get_memory_usage()
        gpu_stats = get_gpu_memory_usage()
        
        memory_critical = mem_stats["ram_percent"] > self.max_memory_percentage
        
        gpu_critical = False
        if gpu_stats and "gpu_percent" in gpu_stats:
            gpu_critical = gpu_stats["gpu_percent"] > self.max_gpu_percentage
        
        return memory_critical, gpu_critical
    
    def unload_least_important_model(self, force_gpu: bool = False) -> bool:
        """
        Unload the least important model based on priority and last use time.
        
        Args:
            force_gpu: If True, only unload GPU models
            
        Returns:
            bool: True if a model was unloaded, False otherwise
        """
        if not self.loaded_models:
            return False
        
        import time
        current_time = time.time()
        
        # Sort models by priority (ascending) and then by last used time (ascending)
        # Only consider GPU models if force_gpu is True
        candidate_models = [
            (model_id, info) for model_id, info in self.loaded_models.items()
            if not force_gpu or info["uses_gpu"]
        ]
        
        if not candidate_models:
            return False
        
        sorted_models = sorted(
            candidate_models,
            key=lambda x: (x[1]["priority"], current_time - x[1]["last_used"])
        )
        
        # Unload the least important model
        model_id, model_info = sorted_models[0]
        self.logger.info(f"Unloading model {model_id} to free resources "
                         f"(priority: {model_info['priority']}, "
                         f"size: {model_info['size_gb']:.2f} GB)")
        
        # Delete the model object and unregister
        del model_info["model"]
        self.unregister_model(model_id)
        
        # Force garbage collection
        free_memory(self.logger)
        
        return True
    
    def manage_resources(self) -> None:
        """
        Check resource usage and unload models if necessary.
        """
        memory_critical, gpu_critical = self.check_resource_limits()
        
        if gpu_critical:
            self.logger.warning(f"GPU memory usage exceeds threshold "
                               f"({self.max_gpu_percentage}%). Unloading GPU models...")
            while gpu_critical and self.unload_least_important_model(force_gpu=True):
                _, gpu_critical = self.check_resource_limits()
        
        if memory_critical:
            self.logger.warning(f"Memory usage exceeds threshold "
                               f"({self.max_memory_percentage}%). Unloading models...")
            while memory_critical and self.unload_least_important_model():
                memory_critical, _ = self.check_resource_limits()
                
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status for display in UI.
        
        Returns:
            dict: Dictionary with resource status information
        """
        mem_stats = get_memory_usage()
        gpu_stats = get_gpu_memory_usage()
        
        status = {
            "memory": {
                "used_percent": mem_stats["ram_percent"],
                "used_gb": mem_stats["ram_used_gb"],
                "total_gb": mem_stats["ram_total_gb"],
                "process_gb": mem_stats["process_ram_gb"],
                "critical": mem_stats["ram_percent"] > self.max_memory_percentage
            },
            "models": {
                "loaded_count": len(self.loaded_models),
                "loaded_models": [
                    {
                        "id": model_id,
                        "size_gb": info["size_gb"],
                        "priority": info["priority"],
                        "uses_gpu": info["uses_gpu"]
                    }
                    for model_id, info in self.loaded_models.items()
                ]
            }
        }
        
        if gpu_stats and "gpu_percent" in gpu_stats:
            status["gpu"] = {
                "used_percent": gpu_stats["gpu_percent"],
                "used_gb": gpu_stats["gpu_used_gb"],
                "total_gb": gpu_stats["gpu_total_gb"],
                "cached_gb": gpu_stats["gpu_cached_gb"],
                "critical": gpu_stats["gpu_percent"] > self.max_gpu_percentage
            }
        
        return status