NutriSync U-Net for Depth Estimation
Overview
This project implements a U-Net neural network for monocular depth estimation, specifically designed to replicate the architecture described in the paper "Deep Optics for Monocular Depth Estimation" by the Deep Optics group. The U-Net processes simulated sensor images or all-in-focus images to predict depth maps at the same resolution as the input, enabling NutriSync to potentially enhance AI-driven food scanning for personalized, culturally inclusive meal planning (e.g., identifying ingredients for Halal or vegetarian recipes). The implementation strictly follows the paper’s specifications for use in computational imaging, supporting NutriSync’s goal of leveraging AI in a timely, inclusive manner within the $6.13B nutrition app market.
Purpose and Context
The U-Net is integrated with a differentiable image formation model that simulates depth-dependent blur, as outlined in the paper. It aims to estimate depth from single images, addressing gaps in competitors like Noom and Mealime, which lack advanced visual processing for diverse dietary needs (25% of feedback values cultural inclusivity). By predicting precise depth maps, NutriSync can enhance AI-driven features like food recognition, aligning with user demands for tailored nutrition plans (61% value AI personalization).
U-Net Architecture
The U-Net follows the paper’s exact design:

Input: RGB images (3 channels, e.g., simulated sensor images or all-in-focus dataset images).
Structure:
5 Downsampling Layers: Each consists of two Conv-BN-ReLU blocks (3x3 convolutions, batch normalization, ReLU activation) followed by 2x2 MaxPooling to reduce spatial dimensions.
5 Upsampling Layers: Each uses a ConvTranspose (2x2, stride 2) for upsampling, concatenates skip connections from the corresponding downsampling layer, and applies two Conv-BN-ReLU blocks.
Output: A single-channel depth map at the input resolution, predicting depth for each pixel.


Channel Progression: Starts at 64 channels, doubles per downsampling layer to 2048 at the bottleneck, then halves back to 64 (inferred from standard U-Net practices, as the paper doesn’t specify counts).
Loss and Training: Uses mean-square-error (MSE) loss on logarithmic depth, trained with the ADAM optimizer for 40,000 iterations, as specified. For specific datasets (e.g., Rectangles), the learning rate decays, but this code focuses on the network structure.

Reference to the Paper
This implementation directly adheres to the Deep Optics for Monocular Depth Estimation paper:

Architecture: Matches the described U-Net with 5 downsampling layers ({Conv-BN-ReLU} × 2 → MaxPool 2×2) and 5 upsampling layers (ConvTranspose + Concat → {Conv-BN-ReLU} × 2), ensuring identical layer structure and skip connections.
Input/Output: Designed to handle RGB inputs and produce depth maps, as used with the paper’s Rectangles, NYU Depth v2, and KITTI datasets.
Integration: Prepared to work with the paper’s differentiable image formation model (though not implemented here), simulating depth-dependent PSFs for computational imaging, critical for NutriSync’s potential food scanning features.

Notes

The U-Net is designed for flexibility with input sizes but assumes proper padding for skip connections (handled automatically in the code).

