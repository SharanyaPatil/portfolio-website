import os
import argparse
import cv2
import numpy as np
from torchvision.transforms import ToPILImage
import torch
from time import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import kornia.morphology as morph
from PIL import Image
try:
    from tqdm import trange
except:
    trange = range
import math
from scipy import ndimage
import gzip
import torch
import pandas as pd

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    # calculate mu_alpha weight
    weight = (1/(K-T_a_L-T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s   = int(T_a_L+1)
    e   = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel-mu), 2)
    return val/len(x)
def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size

    # Print k1 and k2 for debugging
    #print("k1:", k1, "type:", type(k1))
    #print("k2:", k2, "type:", type(k2))
    
    # weight
    w = 2./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    #print("Initial blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Initial blocksize_x:", blocksize_x, "type:", type(blocksize_x))

    # Ensure blocksize_y, blocksize_x, k1, and k2 are integers
    blocksize_y = int(blocksize_y)
    blocksize_x = int(blocksize_x)
    k1 = int(k1)
    k2 = int(k2)

    # Print converted values and types for debugging
    #print("Converted blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Converted blocksize_x:", blocksize_x, "type:", type(blocksize_x))
    #print("Converted k1:", k1, "type:", type(k1))
    #print("Converted k2:", k2, "type:", type(k2))

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # Print the shape of x for debugging
    #print("Shape of x:", x.shape)
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            # bound checks, can't do log(0)
            if min_ == 0.0: val += 0
            elif max_ == 0.0: val += 0
            else: val += math.log(max_/min_)
    return w*val

def _uism(x):
    #print("Shape of input image:", x.shape)
    R = x[:,:,0]
    G = x[:,:,1]
    B = x[:,:,2]
    #print("R channel shape:", R.shape)
    #print("G channel shape:", G.shape)
    #print("B channel shape:", B.shape)

    # Apply Sobel filter
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)
    #print("Sobel R shape:", Rs.shape)
    #print("Sobel G shape:", Gs.shape)
    #print("Sobel B shape:", Bs.shape)

    # Multiply edges by channels
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    #print("R_edge_map shape:", R_edge_map.shape)
    #print("G_edge_map shape:", G_edge_map.shape)
    #print("B_edge_map shape:", B_edge_map.shape)

    # Get EME for each channel
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    #print("r_eme:", r_eme)
    #print("g_eme:", g_eme)
    #print("b_eme:", b_eme)

    # Coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144
    return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

def plip_g(x,mu=1026.0):
    return mu-x

def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k*((g1-g2)/(k-g2))

def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1+g2-((g1*g2)/(gamma))

def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g/gamma) ), c) )

def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    #return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))

def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));

def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta   = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)

def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/abstract/document/5609219
    """
    plip_lambda = 1026.0
    plip_gamma  = 1026.0
    plip_beta   = 1.0
    plip_mu     = 1026.0
    plip_k      = 1026.0
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1]/window_size
    k2 = x.shape[0]/window_size
    #print("k1:", k1, "type:", type(k1))
    #print("k2:", k2, "type:", type(k2))
    k1 = int(k1)
    k2 = int(k2)
    
    # weight
    w = -1./(k1*k2)
    blocksize_x = window_size
    blocksize_y = window_size
    #print("Initial blocksize_y:", blocksize_y, "type:", type(blocksize_y))
    #print("Initial blocksize_x:", blocksize_x, "type:", type(blocksize_x))
    blocksize_y = int(blocksize_y)
    blocksize_x = int(blocksize_x)
    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y*k2, :blocksize_x*k1]
    # entropy scale - higher helps with randomness
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: val += 0.0
            else: val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
            #try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w*val

def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### UCIQE: https://ieeexplore.ieee.org/abstract/document/7300447
    #c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### UIQM https://ieeexplore.ieee.org/abstract/document/7305804
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm,uicm,uism,uiconm

def getUCIQE(img):
    img_BGR = cv2.imread(img)
    img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB) 
    img_LAB = np.array(img_LAB,dtype=np.float64)
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    coe_Metric = [0.4680, 0.2745, 0.2576]
    
    img_lum = img_LAB[:,:,0]/255.0
    img_a = img_LAB[:,:,1]/255.0
    img_b = img_LAB[:,:,2]/255.0

    # item-1
    chroma = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum)*0.99)]
    bottom_index = sorted_index[int(len(img_lum)*0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
    avg_sat = np.mean(sat)

    uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    return uciqe

def measure_UIQMs(dir_name, file_ext=None):
    """
      # measured in RGB
      Assumes:
        * dir_name contain generated images 
        * to evaluate on all images: file_ext = None 
        * to evaluate images that ends with "_SESR.png" or "_En.png"  
            * use file_ext = "_SESR.png" or "_En.png" 
    """
    paths = sorted(glob(join(dir_name, "*.*")))
    if file_ext:
        paths = [p for p in paths if p.endswith(file_ext)]
    uqims = []
    for img_path in paths:
        im = Image.open(img_path).resize((640,640))
        uqims.append(getUIQM(np.array(im)))
    return np.array(uqims)


class paired_rgb_depth_dataset(Dataset):
    def __init__(self, image_path, depth_path, openni_depth, mask_max_depth, image_height, image_width, device):
        self.image_dir = image_path
        self.depth_dir = depth_path
        self.image_files = sorted(os.listdir(image_path))
        self.depth_files = sorted(os.listdir(depth_path))
        self.device = device
        self.openni_depth = openni_depth
        self.mask_max_depth = mask_max_depth
        self.crop = (0, 0, image_height, image_width)
        self.depth_perc = 0.0001
        self.kernel = torch.ones(3, 3).to(device=device)
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.crop[2], self.crop[3]), transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.PILToTensor(),
        ])
        assert len(self.image_files) == len(self.depth_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        fname = self.image_files[index]
        image = Image.open(os.path.join(self.image_dir, fname))
        # image=image.convert('L')
        depth_fname = self.depth_files[index]
        depth = Image.open(os.path.join(self.depth_dir, depth_fname))
        if depth.mode != 'L':
            depth = depth.convert('L')
        depth_transformed: torch.Tensor = self.image_transforms(depth).float().to(device=self.device)
        if self.openni_depth:
            depth_transformed = depth_transformed / 1000.
        if self.mask_max_depth:
            depth_transformed[depth_transformed == 0.] = depth_transformed.max()
        low, high = torch.nanquantile(depth_transformed, self.depth_perc), torch.nanquantile(depth_transformed,
                                                                                             1. - self.depth_perc)
        
        depth_transformed[(depth_transformed < low) | (depth_transformed > high)] = 0.
        depth_transformed = torch.squeeze(morph.closing(torch.unsqueeze(depth_transformed, dim=0), self.kernel), dim=0)
        left_transformed: torch.Tensor = self.image_transforms(image).to(device=self.device) / 255.
        return left_transformed, depth_transformed, [fname]


def color_constancy_loss(image):
    """
    Computes the Color Constancy Loss for an input image.

    Args:
    - image: Tensor of shape (B, C, H, W) where B = batch size, 
             C = number of channels (3 for RGB), H = height, W = width.

    Returns:
    - loss: Color Constancy Loss (scalar).
    """
    # Ensure the image has 3 channels (RGB)
    if image.shape[1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    # Calculate mean for each color channel
    mu_R = torch.mean(image[:, 0, :, :], dim=(1, 2))  # Mean of Red channel
    mu_G = torch.mean(image[:, 1, :, :], dim=(1, 2))  # Mean of Green channel
    mu_B = torch.mean(image[:, 2, :, :], dim=(1, 2))  # Mean of Blue channel

    # Compute the loss components
    loss_RG = ((mu_R - mu_G) / (mu_R + mu_G + 1e-6))**2
    loss_GB = ((mu_G - mu_B) / (mu_G + mu_B + 1e-6))**2
    loss_BR = ((mu_B - mu_R) / (mu_B + mu_R + 1e-6))**2

    # Sum the losses and take the mean over the batch
    loss = torch.mean(loss_RG + loss_GB + loss_BR)
    
    return loss

    

# class BackscatterNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.backscatter_conv = nn.Conv2d(1, 3, 1, bias=False)
#         # self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)
#         # nn.init.uniform_(self.backscatter_conv.weight, 0, 5)
#         # nn.init.uniform_(self.residual_conv.weight, 0, 5)
#                 # Define two convolution layers for B_inf
#         # self.B_inf_layers = nn.Sequential(
#         #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
#         #     nn.ReLU(),
#         #     nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
#         # )
#         self.B_inf= nn.Conv2d(3, 3, 1, bias=False) #for 1 layer
#         #         # Initialize weights for both layers
#         # for layer in self.B_inf_layers:
#         #     if isinstance(layer, nn.Conv2d):
#         #         nn.init.uniform_(layer.weight, 0, 5)

#         nn.init.uniform_(self.B_inf.weight, 0, 5) #1 layer
#         # self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
#         #self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
#         self.sigmoid = nn.Sigmoid()
#         self.tanh=nn.Tanh()
#         self.relu = nn.ReLU()

#     def forward(self, image, depth):
#         # beta_b_conv = self.relu(self.backscatter_conv(depth))
#         # beta_d_conv = self.relu(self.residual_conv(depth))
#         B_inf = self.relu(self.B_inf(image)) #for 1 layer

#         # B_inf = self.B_inf_layers(image)
#         # Bc = self.B_inf * (1 - torch.exp(-beta_b_conv)) #+ self.J_prime * torch.exp(-beta_d_conv)
#         #backscatter = self.sigmoid(B_inf)
#         backscatter = self.tanh(B_inf)
#         #backscatter_masked = backscatter * (depth > 0.).repeat(1, 3, 1, 1)
#         direct = image - backscatter
#         #direct = image - B_inf
#         div= (direct/ depth)
#         J = backscatter+ div
#         return B_inf,backscatter,J,direct,div
    

class BackscatterNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define separate layers for each channel
        self.R_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.G_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.B_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # #2nd layer
        # self.R_layer2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.G_layer2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.B_layer2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize weights
        nn.init.uniform_(self.R_layer.weight, 0, 5)
        nn.init.uniform_(self.G_layer.weight, 0, 5)
        nn.init.uniform_(self.B_layer.weight, 0, 5)
        
        # # Initialize weights
        # nn.init.uniform_(self.R_layer2.weight, 0, 5)
        # nn.init.uniform_(self.G_layer2.weight, 0, 5)
        # nn.init.uniform_(self.B_layer2.weight, 0, 5)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self, image, depth):
        # Separate the R, G, B channels
        R = image[:, 0:1, :, :]
        G = image[:, 1:2, :, :]
        B = image[:, 2:3, :, :]
        
        # Enhance each channel separately
        R_enhanced = self.relu(self.R_layer(R))
        G_enhanced = self.relu(self.G_layer(G))
        B_enhanced = self.relu(self.B_layer(B))
        
        # R_enhanced2 = self.relu(self.R_layer2(R_enhanced))
        # G_enhanced2 = self.relu(self.G_layer2(G_enhanced))
        # B_enhanced2 = self.relu(self.B_layer2(B_enhanced))

        # Combine the enhanced channels back into a single image
        B_inf = torch.cat([R_enhanced, G_enhanced, B_enhanced], dim=1)
        # B_inf = torch.cat([R_enhanced2, G_enhanced2, B_enhanced2], dim=1)
        
        # Backscatter computation
        backscatter = self.tanh(B_inf)
        # direct = image - B_inf
        direct = image - backscatter
        # Dividing by depth
        div = direct / depth
        
        # Final enhanced image
        J = backscatter + div
        return B_inf, backscatter,J,direct, div


# class DeattenuateNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.attenuation_conv = nn.Conv2d(1, 6, 1, bias=False)
#         nn.init.uniform_(self.attenuation_conv.weight, 0, 5)
#         self.attenuation_coef = nn.Parameter(torch.rand(6, 1, 1))
#         self.relu = nn.ReLU()
#         self.wb = nn.Parameter(torch.rand(1, 1, 1))
#         nn.init.constant_(self.wb, 1)
#         self.output_act = nn.Sigmoid()

#     def forward(self, direct, depth):
#         attn_conv = torch.exp(-self.relu(self.attenuation_conv(depth)))
#         beta_d = torch.stack(tuple(
#             torch.sum(attn_conv[:, i:i + 2, :, :] * self.relu(self.attenuation_coef[i:i + 2]), dim=1) for i in
#             range(0, 6, 2)), dim=1)
#         f = torch.exp(torch.clamp(beta_d * depth, 0, float(torch.log(torch.tensor([3.])))))
#         f_masked = f * ((depth == 0.) / f + (depth > 0.))
#         J = f_masked * direct * self.wb
#         nanmask = torch.isnan(J)
#         if torch.any(nanmask):
#             print("Warning! NaN values in J")
#             J[nanmask] = 0
#         return f_masked, J


# class BackscatterLoss(nn.Module):
#     def __init__(self, cost_ratio=1000.):
#         super().__init__()
#         self.l1 = nn.L1Loss()
#         self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
#         self.mse = nn.MSELoss()
#         self.relu = nn.ReLU()
#         self.cost_ratio = cost_ratio

#     def forward(self, B_inf):
#         pos = self.l1(self.relu(B_inf), torch.zeros_like(B_inf))
#         neg = self.smooth_l1(self.relu(-B_inf), torch.zeros_like(B_inf))
#         bs_loss = self.cost_ratio * neg + pos
#         return bs_loss

def histogram_equalization_loss(image, num_bins=256):
    """
    Compute histogram equalization loss to encourage uniform histogram distribution
    Args:
        image: Input tensor of shape (B, C, H, W)
        num_bins: Number of histogram bins
    Returns:
        Scalar loss value
    """
    batch_size, channels, height, width = image.shape
    loss = 0
    
    # Process each channel separately
    for c in range(channels):
        # Get current channel
        channel = image[:, c:c+1, :, :]
        
        # Calculate histogram for this channel
        hist = torch.histc(channel, bins=num_bins, min=0, max=1)
        hist = hist / (height * width)  # Normalize histogram
        
        # Target uniform distribution
        target_hist = torch.ones_like(hist) / num_bins
        
        # Calculate KL divergence between actual and target histogram
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        hist = hist + eps
        target_hist = target_hist + eps
        
        # KL divergence loss
        loss += torch.sum(target_hist * torch.log(target_hist / hist))
        
    return loss / channels



class BackscatterLoss(nn.Module):
    def __init__(self, cost_ratio=1000.):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def acdl_loss(self, enhanced, ground_truth, num_patches=8):
        B, C, H, W = enhanced.shape
        patch_size = H // 10  # Dynamic patch size
        patches_enhanced = []
        patches_ground_truth = []
    
        for _ in range(num_patches):
            x = torch.randint(0, H - patch_size, (1,)).item()
            y = torch.randint(0, W - patch_size, (1,)).item()
            patches_enhanced.append(enhanced[:, :, x:x+patch_size, y:y+patch_size])
            patches_ground_truth.append(ground_truth[:, :, x:x+patch_size, y:y+patch_size])
    
        patches_enhanced = torch.stack(patches_enhanced, dim=1)
        patches_ground_truth = torch.stack(patches_ground_truth, dim=1)
    
        Fg = patches_enhanced - torch.mean(patches_enhanced, dim=(3, 4), keepdim=True)
        Fc = patches_ground_truth - torch.mean(patches_ground_truth, dim=(3, 4), keepdim=True)
    
        dk = Fg - Fc
        mu = torch.mean(dk, dim=(2, 3, 4), keepdim=True)
        sigma = torch.std(dk, dim=(2, 3, 4), keepdim=True) + 1e-6
        dk_normalized = (dk - mu) / sigma
        lossacdl=-torch.mean(torch.log(torch.abs(dk_normalized) + 1e-6))
    
        return lossacdl
    
    def ucar_loss(self, enhanced):
        mean_rgb = torch.mean(enhanced, dim=(2, 3), keepdim=True)
        target = torch.mean(mean_rgb, dim=1, keepdim=True)
        color_loss = torch.mean((mean_rgb - target) ** 2)
        return color_loss



    
    def sobel_edge_loss(self, I_pred, I_true):
        sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])
        sobel_y = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])
    
    
        # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
        I_pred = I_pred.squeeze().cpu().numpy()

        #    If the image is still 4D after squeezing, handle it
        if I_pred.ndim == 4:
            # Typically, we take the first image in the batch and the first channel
            I_pred = I_pred[0, 0]  # This will give you the height x width part, reducing to 2D

        #        If the image is 3D (like RGB), convert it to grayscale
        if I_pred.ndim == 3 and I_pred.shape[0] == 3:
            I_pred = np.mean(I_pred, axis=0)  # Convert to grayscale by averaging across channels

        # Ensure the result is 2D
        I_pred = np.squeeze(I_pred)

        #I_true = I_true.cpu().numpy()
        
        # Assuming 'img' is a 4D tensor in the form (batch_size, channels, height, width)
        I_true = I_true.squeeze().cpu().numpy()

        #    If the image is still 4D after squeezing, handle it
        if I_true.ndim == 4:
            # Typically, we take the first image in the batch and the first channel
            I_true = I_true[0, 0]  # This will give you the height x width part, reducing to 2D

        #        If the image is 3D (like RGB), convert it to grayscale
        if I_true.ndim == 3 and I_true.shape[0] == 3:
            I_true = np.mean(I_true, axis=0)  # Convert to grayscale by averaging across channels

        # Ensure the result is 2D
        I_true = np.squeeze(I_true)

        if I_pred.ndim == 4:
            I_pred = I_pred[0]
            I_true = I_true[0]
    
        loss_total = 0

        for c in range(I_pred.shape[0]):
            sobel_x_pred = cv2.filter2D(I_pred[c], -1, sobel_x)
            sobel_x_true = cv2.filter2D(I_true[c], -1, sobel_x)
            sobel_y_pred = cv2.filter2D(I_pred[c], -1, sobel_y)
            sobel_y_true = cv2.filter2D(I_true[c], -1, sobel_y)
        
            loss_x = np.abs(sobel_x_pred - sobel_x_true)
            loss_y = np.abs(sobel_y_pred - sobel_y_true)
        
            loss_total += np.mean(loss_x + loss_y)
    
        return loss_total / I_pred.shape[0]

    # def forward(self, B_inf):
    #     channel_intensities = torch.mean(B_inf, dim=[2, 3], keepdim=True)
       
    #     bs_loss = (channel_intensities - self.target_intensity).square().mean()
    #     GrayWorldLoss:
    def forward(self,backscatter,B_inf,J):
        saturation_loss = (self.relu(-B_inf) + self.relu(B_inf - 1)).square().mean()
        mean_color = B_inf.mean(dim=(2, 3), keepdim=True)  # Average across spatial dimensions
        gray_level = mean_color.mean(dim=1, keepdim=True)  # Average across RGB channels
        gw_loss = (mean_color - gray_level).abs().mean()

        Y_pred = J.cpu().detach()
        Y_true = B_inf.cpu().detach()
        hist_loss = histogram_equalization_loss(B_inf)
        sobel_loss_value = self.sobel_edge_loss(Y_pred, Y_true)

        acdl_loss_value = self.acdl_loss(backscatter, B_inf)
        ucar_loss_value = self.ucar_loss(backscatter)

        channel_intensities = torch.mean(backscatter, dim=[2, 3], keepdim=True)
        int_loss = (channel_intensities - self.target_intensity).square().mean()
        cc_loss = color_constancy_loss(B_inf)
        return gw_loss#gw_loss
        #((0.5*hist_loss) + (0.5*cc_loss)+(0.5*saturation_loss))
        # #bs_loss+saturation_loss + ucar_loss_value
        # #cc_loss#loss
        # #+bs_loss+ loss+

        # return bs_loss

# class DeattenuateLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.relu = nn.ReLU()
#         self.target_intensity = 0.5

#     def forward(self, direct, J):
#         saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
#         init_spatial = torch.std(direct, dim=[2, 3])
#         channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
#         channel_spatial = torch.std(J, dim=[2, 3])
#         intensity_loss = (channel_intensities - self.target_intensity).square().mean()
#         spatial_variation_loss = self.mse(channel_spatial, init_spatial)
#         if torch.any(torch.isnan(saturation_loss)):
#             print("NaN saturation loss!")
#         if torch.any(torch.isnan(intensity_loss)):
#             print("NaN intensity loss!")
#         if torch.any(torch.isnan(spatial_variation_loss)):
#             print("NaN spatial variation loss!")
#         return saturation_loss + intensity_loss + spatial_variation_loss
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def main(args):
    uciqe_values = []
    uqims_values = []
    uicm_values = []
    uism_values = []
    uicomn_values = []
    output_names = []
    seed = int(torch.randint(9223372036854775807, (1,))[0]) if args.seed is None else args.seed
    if args.seed is None:
        print('Seed:', seed)
    torch.manual_seed(seed)
    torch.autograd.set_detect_anomaly(True)

    train_dataset = paired_rgb_depth_dataset(args.images, args.depth, args.depth_16u, args.mask_max_depth, args.height,
                                             args.width, args.device)
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    check_dir = args.checkpoints
    os.makedirs(check_dir, exist_ok=True)
    target_batch_size = args.batch_size
    dataloader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=False)
    bs_model = BackscatterNet().to(device=args.device)
    
    bs_criterion = BackscatterLoss().to(device=args.device)
    
    bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
  
    skip_right = True
    total_bs_eval_time = 0.
    total_bs_evals = 0
    total_at_eval_time = 0.
    total_at_evals = 0
    # os.makedirs(save_dir, exist_ok=True)
    # check_dir = args.checkpoints
    # os.makedirs(check_dir, exist_ok=True)
    # target_batch_size = args.batch_size
    # dataloader = DataLoader(train_dataset, batch_size=target_batch_size, shuffle=False)
    # bs_model = BackscatterNet().to(device=args.device)
    # # da_model = DeattenuateNet().to(device=args.device)
    # bs_criterion = BackscatterLoss().to(device=args.device)
    # # da_criterion = DeattenuateLoss().to(device=args.device)
    # bs_optimizer = torch.optim.Adam(bs_model.parameters(), lr=args.init_lr)
    # # da_optimizer = torch.optim.Adam(da_model.parameters(), lr=args.init_lr)
    skip_right = True
    total_bs_eval_time = 0.
    total_bs_evals = 0
    total_at_eval_time = 0.
    total_at_evals = 0
    for j, (left, depth, fnames) in enumerate(dataloader):
        print("training")
        bs_model.apply(reset_weights)
        image_batch = left
        batch_size = image_batch.shape[0]
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            B_inf,backscatter,J,direct,div = bs_model(image_batch, depth)
            bs_loss = bs_criterion(backscatter,B_inf,J)
            bs_optimizer.zero_grad()
            bs_loss.backward()
            bs_optimizer.step()
            total_bs_eval_time += time() - start
            total_bs_evals += batch_size
        direct_mean = direct.mean(dim=[2, 3], keepdim=True)
        direct_std = direct.std(dim=[2, 3], keepdim=True)
        direct_z = (direct - direct_mean) / direct_std
        clamped_z = torch.clamp(direct_z, -5, 5)
        direct_no_grad = torch.clamp(
            (clamped_z * direct_std) + torch.maximum(direct_mean, torch.Tensor([1. / 255]).to(device=args.device)), 0, 1).detach()
        for iter in trange(args.init_iters if j == 0 else args.iters):  # Run first batch for 500 iters, rest for 50
            start = time()
            # f, J = da_model(direct_no_grad, depth)
            # da_loss = da_criterion(direct_no_grad, J)
            # da_optimizer.zero_grad()
            # da_loss.backward()
            # da_optimizer.step()
            total_at_eval_time += time() - start
            total_at_evals += batch_size
        # print("Losses: %.9f %.9f" % (bs_loss.item(), da_loss.item()))
        avg_bs_time = total_bs_eval_time / total_bs_evals * 1000
        avg_at_time = total_at_eval_time / total_at_evals * 1000
        avg_time = avg_bs_time + avg_at_time
        print("Avg time per eval: %f ms (%f ms bs, %f ms at)" % (avg_time, avg_bs_time, avg_at_time))
        img = image_batch.cpu()
        direct_img = torch.clamp(direct_no_grad, 0., 1.).cpu()
        B_inf_img = torch.clamp(B_inf, 0., 1.).cpu()
        
        backscatter_img = torch.clamp(backscatter, 0., 1.).detach().cpu()
        # f_img = f.detach().cpu()
        # f_img = f_img / f_img.max()
        # J_img = torch.clamp(J, 0., 1.).cpu()
        # # Convert PIL image to a NumPy array
        # J_corr= ToPILImage(J_img)
        # J_img_np = np.array(J_corr)

        # Assuming J_img is a 4D Tensor with shape (batch_size, channels, height, width)
        J_img = torch.clamp(J, 0., 1.).cpu()
        div_img = torch.clamp(div, 0., 1.).cpu()

        # # If J_img has 4 dimensions (batch size, channels, height, width), select the first image
        # J_img = J_img[0]  # Select the first image from the batch

        # # Convert the Tensor to a PIL Image
        # J_corr = ToPILImage()(J_img)

        # # Convert PIL image to NumPy array
        # J_img_np = np.array(J_corr)

        # # Ensure the NumPy array is in uint8 format (OpenCV expects uint8)
        # J_img_np = (J_img_np * 255).astype(np.uint8)

        # # Check if the image is already RGB (3 channels), avoid conversion if true
        # if len(J_img_np.shape) == 3 and J_img_np.shape[2] == 3:
        #     J_img_rgb = J_img_np  # Already in RGB format, no conversion needed
        # else:
        #     # Convert from grayscale to RGB (only if needed)
        #     J_img_rgb = cv2.cvtColor(J_img_np, cv2.COLOR_GRAY2RGB)
        # J_img_rgb_tensor = torch.from_numpy(J_img_rgb).permute(2, 0, 1).float() / 255.0

        for side in range(1 if skip_right else 2):
            side_name = 'left' if side == 0 else 'right'
            names = fnames[side]
            for n in range(batch_size):
                i = n + target_batch_size * side
                #if args.save_intermediates:
                #save_image(direct_img[i], "%s/%s-direct.png" % (save_dir, names[n].rstrip('.png')))
                save_image(B_inf[i], "%s/%s-Binf.png" % (save_dir, names[n].rstrip('.png')))
                # save_image(B_inf_img[i], "%s/%s-Binfimg.png" % (save_dir, names[n].rstrip('.png')))
                save_image(backscatter_img[i], "%s/%s-backscatter.png" % (save_dir, names[n].rstrip('.png')))
                    # save_image(f_img[i], "%s/%s-f.png" % (save_dir, names[n].rstrip('.png')))
                #save_image(div_img[i], "%s/%s-div.png" % (save_dir, names[n].rstrip('.png')))
                save_image(J_img[i], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                output_image_path ="%s/%s-backscatter.png" % (save_dir, names[n].rstrip('.png'))
                output_image =Image.open(output_image_path)
                output_image = output_image.resize((256, 256))
                image = output_image.convert('RGB')
                image_array = np.array(image)
                # if J_img_rgb_tensor.dim() == 4:
                #     save_image(J_img_rgb_tensor[], "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                # # else:
                # #     # If J_img_rgb_tensor is 3D (single image), you can directly use it
                # #     save_image(J_img_rgb_tensor, "%s/%s-corrected.png" % (save_dir, names[n].rstrip('.png')))
                uciqe_value = getUCIQE(output_image_path)
                print('UCIQE:',uciqe_value)
                uqims_value,uicm_value,uism_value,uicomn_value =getUIQM(image_array)
                print('UQIMS:',uqims_value)
                uciqe_values.append(uciqe_value)
                uqims_values.append(uqims_value)
                uicm_values.append(uicm_value)
                uism_values.append(uism_value)
                uicomn_values.append(uicomn_value)
                output_names.append(names[n])

        # Save checkpoint with compression
        checkpoint_path = os.path.join(check_dir, f'model_checkpoint_{j}.pth')
        with gzip.open(checkpoint_path, 'wb') as f:
            torch.save({
            'bs_model_state_dict': bs_model.state_dict(),
            #'da_model_state_dict': da_model.state_dict(),
            'bs_optimizer_state_dict': bs_optimizer.state_dict(),
            #'da_optimizer_state_dict': da_optimizer.state_dict(),
            }, f)
    df = pd.DataFrame({
        'Output Image Name': output_names,
        'uciqe': uciqe_values,
        'uqims': uqims_values,
        'uicm':uicm_values,
        'uism':uism_values,
        'uicomn':uicomn_values
        })
    
    excel_path = os.path.join(save_dir, 'evaluation_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Evaluation metrics saved to {excel_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--images', type=str, default='/home/user/Oceanlens/images_seethruWBGC14', help='Path to the images folder')
    parser.add_argument('--images', type=str, default='/media/user/One Touch/Oceanlens/backup/Oceanlens/Images_seethru_raw_no_berman', help='Path to the images folder')
    #parser.add_argument('--depth', type=str, default='/home/user/Oceanlens/depth_depthanyseathru' , help='Path to the depth folder')
    parser.add_argument('--depth', type=str, default='/media/user/One Touch/Oceanlens/backup/Oceanlens/Images_seethru_raw_no_berman-newtransdepth' , help='Path to the depth folder')
    #parser.add_argument('--depth', type=str, default='/home/user/Oceanlens/SFM_depth_images' , help='Path to the depth folder')
    parser.add_argument('--output', type=str, default=f'/media/user/One Touch/Revesesea/new/output_trans_tanh_binf_gw_bs_raw_3_ch_1',  help='Path to the output folder')
    parser.add_argument('--checkpoints', type = str, default= r'/media/user/One Touch/Revesesea/new/output_trans_tanh_binf_gw_bs_raw_3_ch_1/check')
    parser.add_argument('--height', type=int, default=1242, help='Height of the image and depth files')
    parser.add_argument('--width', type=int, default=1952, help='Width of the image and depth')
    parser.add_argument('--depth_16u', action='store_true',
                        help='True if depth images are 16-bit unsigned (millimetres), false if floating point (metres)')
    parser.add_argument('--mask_max_depth', action='store_true',
                        help='If true will replace zeroes in depth files with max depth')
    parser.add_argument('--seed', type=int, default=None, help='Seed to initialize network weights (use 1337 to replicate paper results)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing images')
    parser.add_argument('--save_intermediates', action='store_true', default=False, help='Set to True to save intermediate files (backscatter, attenuation, and direct images)')
    parser.add_argument('--init_iters', type=int, default=150, help='How many iterations to refine the first image batch (should be >= iters)')
    parser.add_argument('--iters', type=int, default=150, help='How many iterations to refine each image batch')
    parser.add_argument('--init_lr', type=float, default=1e-2, help='Initial learning rate for Adam optimizer')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    main(args)






    
