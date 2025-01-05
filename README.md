# BRepNet Masking Modification Experiment

This repository is made for the purposes of the T.U.M course SoftwareLab 2024, part of the project "AI Supported automated feature Recognition". In the spirit of having a first contact and understanding a NN focused on feature recognition and with the goal of learning how SSL may be implementaed, the following modifications were made in the BRrepNet model file (brepnet.py). The modifications applied block masking to Xc (coedge feature matrix) and Xf (face features matrix). 

## Overview

The modifications to Xc are as follows after Xc has been created:

Xc_masked = Xc.clone()

masking_ratio = 0.2

mask_indices = None

if self.training and masking_ratio > 0:

  num_coedges = Xc.size(0)
  
  mask_indices = torch.randperm(num_coedges)[:int(masking_ratio * num_coedges)]
  
  mask_indices = mask_indices[mask_indices < num_coedges]
  
  #Block masking
  
  block_size = int(masking_ratio * num_coedges)
  
  start_index = torch.randint(0, num_coedges - block_size + 1, (1,)).item()
  
  Xc_masked[start_index:start_index + block_size] = 0
  
The same is applied to Xf and then Xc_masked and Xf_masked are passed through the netowrk in the same iteration as Xc and Xf. Then then face_embeddings produced by the masked and unmasked matrices are compared with the following loss function: 

mask_loss = F.mse_loss(face_embeddings, face_embeddings_masked, reduction='mean')

total_loss = segmentation_loss  + ssl_weight * mask_loss

## Outcome

This altered version run stably and even produced slightly higher accuracy (~1.5%) and IoU in the first 10 epochs of training. After, 15 to 20 epochs this difference subsides. These changes are similar to dropout, but targeted to those 2 specific matrices, trying to encourage the NN to run on less detailed feature data. 

## Requirements

A working installation of the BRepNet framework. Refer to the official BRepNet repository for installation instructions. https://github.com/AutodeskAILab/BRepNet

Python 3.8+ and the dependencies specified by the original BRepNet repository.

## Setup Instructions

Replace the brepnet.py file in the BRepNet/models directory with the modified version provided in this repository.

## Acknowledgments

This work is heavily based upon the original BRepNet framework found here https://github.com/AutodeskAILab/BRepNet . 
