"""
NDVI visualization adapted for S2-Agri Pixel-Set data.

Original code loaded single-date GeoTIFF rasters.
This version loads a pixel-set sample (shape [T, C, P]) and visualizes:
  - Red band (channel 2) over time
  - NIR band (channel 6) over time
  - NDVI = (NIR - RED) / (NIR + RED) over time

Band indices follow layers.py convention:
  blue=1, red=2, near_infrared=6, swir1=8
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import PixelSetData
def plot_pixel_set_sample(SAMPLE_IDX=0):
  
  DATA_FOLDER = "/home/emilie/Documents/IODAA/classification_site/s2_agri_pixel_set"
  EPS         = 1e-6       # avoid division by zero
  QUANTIF     = 10000.0    # reflectance quantification value

  dataset = PixelSetData(DATA_FOLDER, set='test')
  image, doys, label = dataset[SAMPLE_IDX]   # image: [T, C, P]

  # Reflectance in [0, 1]
  image = image.float() / QUANTIF            # [T, C, P]

  T, C, P = image.shape

  # Extract bands and average over pixels → shape [T]
  RED  = image[:, 2, :].mean(dim=1).numpy()   # [T]
  NIR  = image[:, 6, :].mean(dim=1).numpy()   # [T]
  BLUE = image[:, 1, :].mean(dim=1).numpy()      # [T]
  SWIR1 = image[:, 8, :].mean(dim=1).numpy()   # [T]  
  NDVI = (NIR - RED) / (NIR + RED + EPS)           # [T]
  BI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))

  doys_np = doys.numpy()                            # [T]

  # Remove padded (doy == 0) timesteps
  valid = doys_np > 0
  doys_np = doys_np[valid]
  RED    = RED[valid]
  BLUE  = BLUE[valid]
  SWIR1  = SWIR1[valid]
  NIR    = NIR[valid]
  NDVI   = NDVI[valid]
  BI     = BI[valid]
  # ── Plot ──────────────────────────────────────────────────────────────────────
  fig, axs = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')

  maxi = max(RED.max(), NIR.max())
  clim = [0, maxi * 0.5]   # mirrors original clim=[0, maxi-maxi*0.5]

  # — Red band —
  sc = axs[0].scatter(doys_np, RED, c=RED, cmap='Reds', vmin=clim[0], vmax=clim[1], s=30)
  axs[0].set_title('Rouge (Red band)')
  axs[0].set_xlabel('Day of Year')
  axs[0].set_ylabel('Réflectance')
  plt.colorbar(sc, ax=axs[0], location='bottom', label='réflectance')

  # — NIR band —
  sc2 = axs[1].scatter(doys_np, NIR, c=NIR, cmap='Reds', vmin=clim[0], vmax=clim[1], s=30)
  axs[1].set_title('Infra Rouge (NIR band)')
  axs[1].set_xlabel('Day of Year')
  axs[1].set_ylabel('Réflectance')
  plt.colorbar(sc2, ax=axs[1], location='bottom', label='réflectance')

  # — NDVI —
  sc3 = axs[2].scatter(doys_np, NDVI, c=NDVI, cmap='RdYlGn', vmin=-0.9, vmax=0.9, s=30)
  axs[2].set_title(f'NDVI  —  label: {dataset.label_names[label]}')
  axs[2].set_xlabel('Day of Year')
  axs[2].set_ylabel('NDVI')
  plt.colorbar(sc3, ax=axs[2], location='bottom', label='NDVI')

  # — BI —
  sc4 = axs[3].scatter(doys_np, BI, c=BI, cmap='RdYlGn', vmin=-0.9, vmax=0.9, s=30)
  axs[3].set_title(f'BI  —  label: {dataset.label_names[label]}')
  axs[3].set_xlabel('Day of Year')
  axs[3].set_ylabel('BI')
  plt.colorbar(sc4, ax=axs[3], location='bottom', label='BI')

  fig.suptitle(f'Sample #{SAMPLE_IDX}  —  {T} acquisitions, {P} pixels', fontsize=12)
  plt.savefig('ndvi_pixelset.png', dpi=150, bbox_inches='tight')
  plt.show()