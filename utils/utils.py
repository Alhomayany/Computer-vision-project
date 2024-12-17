import torch
from torch.nn import Module # for type hinting
from ultralytics import YOLO

from utils.rcnn_wrapper import RCNN_YOLO, ModelWrapper

import matplotlib.pyplot as plt

def get_model_size(model: Module | YOLO, verbose=True, name=None) -> dict:
    if isinstance(model, YOLO):
        model = model.model
    if isinstance(model, ModelWrapper):
        model = model.model
    if not isinstance(model, Module):
        raise TypeError("model must be torch.nn.Module or ultralytics.YOLO instance")
    
    params_memory_size = sum(p.element_size() * p.nelement() for p in model.parameters())
    nparams = sum(p.nelement() for p in model.parameters())
    mb = params_memory_size / (1024 ** 2) # to mb
    mb = round(mb, 3) # kb precision
    nparams_millions = nparams / 1e6 # to millions
    name = model.__class__.__name__ if name is None else name
    if verbose:
        print('-' * 30)
        print(f'Model name: {name}')
        print(f"Model size: {mb:.0f} MB")
        print(f"Number of parameters: {nparams_millions:.1f} million")
        print('-' * 30 + '\n')

    return {"mb": mb, "nparams": nparams}

def measure_inference_vram(model: Module | YOLO, name=None, device='cuda', seed=511) -> float:
    """make sure device is already set"""
    if isinstance(model, YOLO):
        model = model.model
    if isinstance(model, ModelWrapper):
        model = model.model
    if not isinstance(model, Module):
        raise TypeError("model must be torch.nn.Module or ultralytics.YOLO instance")
    
    name = model.__class__.__name__ if name is None else name
    # clear cache
    torch.cuda.empty_cache()
    model = model.to(device)
    memory = [torch.cuda.memory_allocated() / 1024**2] # to mb
    
    # get noise tensor
    torch.manual_seed(seed)
    noise = torch.rand(1, 3, 1024, 1024).to(device)
    # add hook to measure memory usage after each layer
    
    def vram_hook(module, input, output):
        memory.append(torch.cuda.memory_allocated() / 1024**2) # to mb
    for layer in model.modules():
        layer.register_forward_hook(vram_hook)
    
    # run inference
    model.eval()
    # clear cache
    torch.cuda.empty_cache()
    with torch.inference_mode():
        model(noise)
    
    peak = (max(memory) - min(memory)) # peak compared to baseline
    peak = round(peak, 3) # kb precision
    return_dict = {"peak": peak}
    return_dict['model_name'] = name
    # build a plot of memory usage
    
    fig, ax = plt.subplots()
    ax.plot(memory)
    ax.set_title(f"Memory usage for {name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("VRAM (MB)")
    ax.grid()

    return_dict['plot'] = fig
    return return_dict

def run_validation(model: YOLO, name=None, yaml_path='data.yaml'):
    if not isinstance(model, YOLO):
        raise TypeError("model must be an instance of ultralytics.YOLO")
    if isinstance(model, RCNN_YOLO):
        name = name if name is not None else model.model.model.__class__.__name__
    else:
        name = name if name is not None else model.__class__.__name__
    print('-' * 30)
    print("Running validation...")
    print(f'Model: {name}')
    metrics = model.val(
        data=yaml_path,
        split='val',
        batch=1,
        #imgsz=640,
        plots=True,  # Generate plots of results,
        save_json=True,
        project='custom_val_output',
        name=name
    )
    get_model_size(model)
    print("Validation complete!")
    
    return metrics
    
