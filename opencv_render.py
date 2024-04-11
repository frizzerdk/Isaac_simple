import cv2
import numpy as np
import torch  # Assuming you're using PyTorch tensors

def render_tensor(tensor):
    # Step 1: Convert the tensor to a NumPy array
    # Ensure tensor is on CPU and convert to numpy
    np_array = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    
    # Step 2: Normalize the tensor to 0-255
    # Assuming the tensor's values range from min to max, normalize to 0-255
    normalized_array = cv2.normalize(np_array, None, alpha=0, beta=255, 
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Step 3: Apply a colormap
    # cv2.applyColorMap supports various colormaps like COLORMAP_JET, COLORMAP_HOT, etc.
    heatmap = cv2.applyColorMap(normalized_array, cv2.COLORMAP_JET)
    
    # Display the heatmap
    cv2.imshow('Heatmap', heatmap)
    cv2.waitKey(1)  # Use a small delay so the window is responsive; adjust as necessary



while True:
    tensor = torch.randn(256, 256) 
    # perlin noise tensor
   
    render_tensor(tensor)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break