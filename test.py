import torch
from projection import projection
from backprojection import backprojection
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(phantom, sinogram, reconstruction, filename):
    """Visualize reconstruction results"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(phantom, cmap='gray')
    plt.title('Original Phantom')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.title('Sinogram')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(reconstruction, cmap='gray')
    plt.title('Reconstruction')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'{filename}.png')

nDetector     = 800 
dDetector     = 1.0
dImage_x      = 512.0
dImage_y      = 512.0
nImage_x      = 256
nImage_y      = 256
nViews        = 720       # Number of views
DSD           = 1100.0    # Distance Source Detector
DSO           = 1000.0    # Distance Source Origin

img2 = torch.from_numpy(np.load('./data/0000.npy')).cuda()
imgP2 = projection(img2, nDetector, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, nViews, DSD, DSO)
rec2 = backprojection(imgP2, dDetector, dImage_x, dImage_y, nImage_x, nImage_y, DSD, DSO)
visualize_results(img2.detach().cpu().numpy(), imgP2.detach().cpu().numpy(), rec2.detach().cpu().numpy(), 'CT_sample1')
