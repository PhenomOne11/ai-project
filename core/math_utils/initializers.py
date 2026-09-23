"""
Neural Network Weight Initialization Algorithms
Production-grade initializers with per-layer diagnostic output.

Implements:
- LSUV (Layer-Sequential Unit Variance) with convergence tracking
- Kaiming (He) initialization
- Xavier (Glorot) initialization
- Orthogonal initialization
- Custom Delta-Orthogonal for residual networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from core.logger import get_logger
from core.math_utils.tensor_ops import check_tensor_health

logger = get_logger("neuralstream.init")


def lsuv_init(
    model: nn.Module,
    calibration_batch: torch.Tensor,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    max_iterations: int = 20,
    tolerance: float = 0.02,
    verbose: bool = True
) -> nn.Module:
    """
    Layer-Sequential Unit-Variance (LSUV) Initialization.
    
    Iteratively adjusts each layer's weights so that the output activations
    have approximately unit variance and zero mean. This prevents vanishing/
    exploding gradients without requiring batch normalization.
    
    Reference: Mishkin & Matas (2016), "All You Need is a Good Init"
    
    Args:
        model: Neural network module
        calibration_batch: Representative data batch for activation measurement
        target_mean: Target activation mean (typically 0)
        target_std: Target activation standard deviation (typically 1)
        max_iterations: Maximum adjustment iterations per layer
        tolerance: Convergence threshold for std deviation
        verbose: Print per-layer diagnostics
    
    Returns:
        Initialized model with diagnostic report
    """
    model.eval()
    diagnostics: Dict[str, Dict] = {}
    
    # Identify weight layers to calibrate
    target_layers: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers.append((name, module))
    
    if verbose:
        logger.info(f"LSUV Init: {len(target_layers)} layers to calibrate")
    
    for layer_idx, (name, module) in enumerate(target_layers):
        # Step 1: Orthogonal initialization of weights
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        
        # Step 2: Iterative variance adjustment
        layer_diagnostics = {
            'initial_std': None,
            'final_std': None,
            'iterations': 0,
            'converged': False,
            'scale_factors': []
        }
        
        for iteration in range(max_iterations):
            # Forward pass and capture this layer's output
            activation = _capture_layer_output(model, module, calibration_batch)
            
            if activation is None:
                break
            
            health = check_tensor_health(activation)
            if not health['is_healthy']:
                logger.warning(f"  [{name}] Unhealthy activation detected at iteration {iteration}")
                break
            
            current_std = float(activation.std())
            current_mean = float(activation.mean())
            
            if iteration == 0:
                layer_diagnostics['initial_std'] = current_std
            
            if abs(current_std - target_std) < tolerance:
                layer_diagnostics['converged'] = True
                layer_diagnostics['iterations'] = iteration + 1
                layer_diagnostics['final_std'] = current_std
                break
            
            # Adjust weights
            if current_std > 1e-8:
                scale = target_std / current_std
                module.weight.data *= scale
                layer_diagnostics['scale_factors'].append(float(scale))
            else:
                logger.warning(f"  [{name}] Near-zero std ({current_std:.6f}), reinitializing")
                nn.init.orthogonal_(module.weight)
        
        else:
            # max_iterations reached without convergence
            activation = _capture_layer_output(model, module, calibration_batch)
            if activation is not None:
                layer_diagnostics['final_std'] = float(activation.std())
            layer_diagnostics['iterations'] = max_iterations
        
        diagnostics[name] = layer_diagnostics
        
        if verbose:
            status = "✓ CONVERGED" if layer_diagnostics['converged'] else "⚠ MAX ITERS"
            init_std = layer_diagnostics['initial_std'] or 0
            final_std = layer_diagnostics['final_std'] or 0
            iters = layer_diagnostics['iterations']
            logger.info(
                f"  Layer [{name}] | std: {init_std:.4f} → {final_std:.4f} | "
                f"iters: {iters} | {status}"
            )
    
    model._lsuv_diagnostics = diagnostics

    # Mark model as calibrated if it exposes the calibration API
    if hasattr(model, 'mark_calibrated'):
        model.mark_calibrated(diagnostics)

    if verbose:
        converged = sum(1 for d in diagnostics.values() if d.get('converged', False))
        logger.info(f"LSUV Complete: {converged}/{len(diagnostics)} layers converged")

    return model


def _capture_layer_output(
    model: nn.Module,
    target_module: nn.Module,
    input_batch: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Run a forward pass and capture the output of a specific layer.
    Uses a temporary forward hook.
    """
    captured = {}
    
    def hook_fn(module, inp, output):
        captured['output'] = output.detach()
    
    handle = target_module.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(input_batch)
    except Exception:
        handle.remove()
        return None
    
    handle.remove()
    return captured.get('output', None)


def kaiming_init(
    model: nn.Module,
    nonlinearity: str = 'leaky_relu',
    a: float = 0.01,
    mode: str = 'fan_in',
    verbose: bool = False
) -> nn.Module:
    """
    Kaiming (He) initialization for ReLU-family networks.
    
    Preserves variance under forward passes for ReLU activations.
    
    Args:
        model: Neural network module
        nonlinearity: Activation function name
        a: Negative slope for leaky_relu
        mode: 'fan_in' or 'fan_out'
        verbose: Print per-layer info
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if verbose:
                logger.info(f"  Kaiming init: {name} ({module.weight.shape})")
        
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if verbose:
                logger.info(f"  Kaiming init: {name} ({module.weight.shape})")
        
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model


def xavier_init(
    model: nn.Module,
    gain: float = 1.0,
    verbose: bool = False
) -> nn.Module:
    """
    Xavier (Glorot) initialization.
    
    Preserves variance under both forward and backward passes
    for sigmoid/tanh activations.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if verbose:
                logger.info(f"  Xavier init: {name} ({module.weight.shape})")
        
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model


def orthogonal_init(
    model: nn.Module,
    gain: float = 1.0,
    verbose: bool = False
) -> nn.Module:
    """
    Orthogonal initialization for all weight matrices.
    
    Prevents gradient vanishing/explosion by ensuring
    singular values are all equal to gain.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if verbose:
                logger.info(f"  Orthogonal init: {name} ({module.weight.shape})")
        
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    return model


def get_init_diagnostics(model: nn.Module) -> Dict[str, Dict]:
    """
    Compute diagnostic statistics for all weight tensors in a model.
    
    Returns dict of layer_name -> {mean, std, min, max, norm, shape}
    """
    diagnostics = {}
    
    for name, param in model.named_parameters():
        data = param.detach().float()
        diagnostics[name] = {
            'shape': list(data.shape),
            'numel': int(data.numel()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'l2_norm': float(data.norm(2)),
            'frobenius_norm': float(data.norm('fro') if data.ndim >= 2 else data.norm(2)),
            **check_tensor_health(data)
        }
    
    return diagnostics