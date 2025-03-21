"""
Resource manager module for handling memory and GPU resources.
"""
from typing import Dict, Any, List, Callable, Optional
import time
import gc
import threading
import torch
import logging

from config.config import MAX_MEMORY_PERCENTAGE, MAX_GPU_MEMORY_PERCENTAGE
from src.utils.logger import setup_logger
from src.utils.resource_monitor import (
    get_memory_usage, get_gpu_memory_usage, log_memory_usage, free_memory
)

logger = setup_logger(__name__, "resource_manager.log")

class ModelResource:
    """
    Class representing a loaded model resource.
    """
    
    def __init__(self, model_id: str, model_obj: Any, model_size_gb: float, 
                uses_gpu: bool, priority: int = 1):
        """
        Initialize model resource.
        
        Args:
            model_id: Unique identifier for the model
            model_obj: The model object
            model_size_gb: Approximate size of the model in GB
            uses_gpu: Whether the model uses GPU
            priority: Priority (higher means more important to keep)
        """
        self.model_id = model_id
        self.model_obj = model_obj
        self.model_size_gb = model_size_gb
        self.uses_gpu = uses_gpu
        self.priority = priority
        self.last_used = time.time()
        self.load_time = time.time()
    
    def update_last_used(self):
        """
        Update the last used timestamp.
        """
        self.last_used = time.time()

class ResourceManager:
    """
    Resource manager for monitoring and managing system resources.
    """
    
    def __init__(self, max_memory_percentage: float = None, 
                max_gpu_percentage: float = None,
                check_interval: int = 60):
        """
        Initialize the resource manager.
        
        Args:
            max_memory_percentage: Maximum memory percentage before unloading models
            max_gpu_percentage: Maximum GPU memory percentage before unloading models
            check_interval: Interval in seconds to check resource usage
        """
        self.max_memory_percentage = max_memory_percentage or MAX_MEMORY_PERCENTAGE
        self.max_gpu_percentage = max_gpu_percentage or MAX_GPU_MEMORY_PERCENTAGE
        self.check_interval = check_interval
        self.models = {}
        self.model_unload_functions = {}
        self.model_load_functions = {}
        self.loading_status = {}
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.running = False
        
        logger.info(f"Initialized ResourceManager with max_memory_percentage={self.max_memory_percentage}, "
                   f"max_gpu_percentage={self.max_gpu_percentage}, check_interval={check_interval}")
    
    def start_monitoring(self):
        """
        Start monitoring resources in a background thread.
        """
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """
        Stop monitoring resources.
        """
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
        
        logger.info("Stopped resource monitoring")
    
    def _monitor_resources(self):
        """
        Monitor resources and unload models if necessary.
        """
        while self.running:
            try:
                self.manage_resources()
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
            
            # Sleep for the check interval
            time.sleep(self.check_interval)
    
    def register_model(self, model_id: str, model_obj: Any, model_size_gb: float, 
                      uses_gpu: bool = True, priority: int = 1,
                      load_func: Optional[Callable] = None,
                      unload_func: Optional[Callable] = None):
        """
        Register a model with the resource manager.
        
        Args:
            model_id: Unique identifier for the model
            model_obj: The model object
            model_size_gb: Approximate size of the model in GB
            uses_gpu: Whether the model uses GPU
            priority: Priority (higher means more important to keep)
            load_func: Function to load the model
            unload_func: Function to unload the model
        """
        with self.lock:
            self.models[model_id] = ModelResource(
                model_id=model_id,
                model_obj=model_obj,
                model_size_gb=model_size_gb,
                uses_gpu=uses_gpu,
                priority=priority
            )
            
            if load_func:
                self.model_load_functions[model_id] = load_func
            
            if unload_func:
                self.model_unload_functions[model_id] = unload_func
            
            self.loading_status[model_id] = "loaded"
        
        logger.info(f"Registered model {model_id} (size: {model_size_gb:.2f} GB, priority: {priority})")
    
    def unregister_model(self, model_id: str):
        """
        Unregister a model from the resource manager.
        
        Args:
            model_id: Unique identifier for the model
        """
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
            
            if model_id in self.model_load_functions:
                del self.model_load_functions[model_id]
            
            if model_id in self.model_unload_functions:
                del self.model_unload_functions[model_id]
            
            if model_id in self.loading_status:
                del self.loading_status[model_id]
        
        logger.info(f"Unregistered model {model_id}")
    
    def mark_model_used(self, model_id: str):
        """
        Mark a model as recently used.
        
        Args:
            model_id: Unique identifier for the model
        """
        with self.lock:
            if model_id in self.models:
                self.models[model_id].update_last_used()
    
    def check_resource_limits(self) -> Dict[str, bool]:
        """
        Check if system is approaching resource limits.
        
        Returns:
            Dictionary with resource status
        """
        mem_stats = get_memory_usage()
        gpu_stats = get_gpu_memory_usage()
        
        memory_critical = mem_stats["ram_percent"] > self.max_memory_percentage
        
        gpu_critical = False
        if gpu_stats and "gpu_percent" in gpu_stats:
            gpu_critical = gpu_stats["gpu_percent"] > self.max_gpu_percentage
        
        return {
            "memory_critical": memory_critical,
            "gpu_critical": gpu_critical,
            "ram_percent": mem_stats["ram_percent"],
            "gpu_percent": gpu_stats.get("gpu_percent", 0.0) if gpu_stats else 0.0
        }
    
    def unload_least_important_model(self, force_gpu: bool = False) -> bool:
        """
        Unload the least important model based on priority and last use time.
        
        Args:
            force_gpu: If True, only unload GPU models
            
        Returns:
            True if a model was unloaded, False otherwise
        """
        with self.lock:
            if not self.models:
                return False
            
            # Sort models by priority (ascending) and then by last used time (ascending)
            # Only consider GPU models if force_gpu is True
            candidate_models = {
                model_id: model for model_id, model in self.models.items()
                if not force_gpu or model.uses_gpu
            }
            
            if not candidate_models:
                return False
            
            # Sort by priority (lower first) and last used time (older first)
            sorted_models = sorted(
                candidate_models.items(),
                key=lambda x: (x[1].priority, x[1].last_used)
            )
            
            # Get the least important model
            model_id, model = sorted_models[0]
            
            # Unload the model
            if self._unload_model(model_id):
                logger.info(f"Unloaded model {model_id} to free resources "
                          f"(priority: {model.priority}, "
                          f"size: {model.model_size_gb:.2f} GB)")
                return True
            
            return False
    
    def _unload_model(self, model_id: str) -> bool:
        """
        Unload a model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models:
            return False
        
        try:
            # Call unload function if available
            if model_id in self.model_unload_functions:
                self.model_unload_functions[model_id]()
            
            # Update status
            self.loading_status[model_id] = "unloaded"
            
            # Remove model object reference
            self.models[model_id].model_obj = None
            
            # Force garbage collection
            free_memory(logger)
            
            return True
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """
        Load a model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.model_load_functions:
            return False
        
        # Check if model is already loaded
        if model_id in self.models and self.models[model_id].model_obj is not None:
            return True
        
        try:
            # Update status
            self.loading_status[model_id] = "loading"
            
            # Call load function
            model_obj = self.model_load_functions[model_id]()
            
            # Update model reference
            if model_id in self.models:
                self.models[model_id].model_obj = model_obj
                self.models[model_id].load_time = time.time()
                self.models[model_id].update_last_used()
            else:
                # Model was unregistered during loading
                return False
            
            # Update status
            self.loading_status[model_id] = "loaded"
            
            logger.info(f"Loaded model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            self.loading_status[model_id] = "error"
            return False
    
    def manage_resources(self):
        """
        Check resource usage and unload models if necessary.
        """
        resource_status = self.check_resource_limits()
        
        log_memory_usage(logger)
        
        # Unload models if resources are critical
        if resource_status["gpu_critical"]:
            logger.warning(f"GPU memory usage exceeds threshold "
                         f"({self.max_gpu_percentage}%). Unloading GPU models...")
            
            while resource_status["gpu_critical"] and self.unload_least_important_model(force_gpu=True):
                resource_status = self.check_resource_limits()
        
        if resource_status["memory_critical"]:
            logger.warning(f"Memory usage exceeds threshold "
                         f"({self.max_memory_percentage}%). Unloading models...")
            
            while resource_status["memory_critical"] and self.unload_least_important_model():
                resource_status = self.check_resource_limits()
    
    def unload_all_models(self):
        """
        Unload all models.
        """
        with self.lock:
            model_ids = list(self.models.keys())
        
        for model_id in model_ids:
            self._unload_model(model_id)
        
        # Force garbage collection
        free_memory(logger)
        
        logger.info("Unloaded all models")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """
        Get current resource status for display in UI.
        
        Returns:
            Dictionary with resource status information
        """
        mem_stats = get_memory_usage()
        gpu_stats = get_gpu_memory_usage()
        
        with self.lock:
            models_info = []
            for model_id, model in self.models.items():
                models_info.append({
                    "id": model_id,
                    "size_gb": model.model_size_gb,
                    "priority": model.priority,
                    "uses_gpu": model.uses_gpu,
                    "last_used": model.last_used,
                    "load_time": model.load_time,
                    "status": self.loading_status.get(model_id, "unknown"),
                    "is_loaded": model.model_obj is not None
                })
        
        status = {
            "memory": {
                "used_percent": mem_stats["ram_percent"],
                "used_gb": mem_stats["ram_used_gb"],
                "total_gb": mem_stats["ram_total_gb"],
                "process_gb": mem_stats["process_ram_gb"],
                "critical": mem_stats["ram_percent"] > self.max_memory_percentage
            },
            "models": {
                "loaded_count": sum(1 for m in models_info if m["is_loaded"]),
                "total_count": len(models_info),
                "models": models_info
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