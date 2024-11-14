Innovative Feature Pyramid Aggregation Network for Robust Facial Expression Recognition in Unconstrained Conditions
=
We propose a feature pyramid aggregation network with multi-head hybrid crisscross attention (MHCA), a simple yet effective network for facial expression recognition that enhances the connectivity of feature maps at different scales in the feature pyramid structure, enhancing global and local attention as well as contextual connections. Our proposed module aggregates information at different scales through an aggregation network, addressing the semantic gap between encoder and decoder features by continuously capturing channel and spatial dependencies across multi-scale encoder features.

 ![Fig 1. The structure of the feature pyramid aggregation block](https://github.com/user-attachments/assets/108f1036-2b49-4110-99e6-d59ad9c8c122)
 
 ![Fig 2. (A) is the overall structure of the MHB, (B) is the channel cross-attention model, (C) is the spatial cross-attention model](https://github.com/user-attachments/assets/246cf8f1-7782-448f-a9e3-102acd5601d2)
 
# The results are as follows:

Table 1. Experimental results of FPAN on FER2013, RAF-DB and Affect Net datasets

Dataset |Accuracy(%) |Recall(%) |Precision(%) |F1-score(%)
----|----|----|-----|----
Fer2013|	75.75|	74.16	|76.65|	75.38
RAF -DB|	88.27|	81.24	|83.52|	82.36
Affect Net|	58.05|	58.04 |58.56|	58.30
# Experimental Implementation
During the training phase, we preprocessed facial expression images by normalizing them to 224×224 pixels, followed by data augmentation. We used the ResNet-34 network for feature extraction, and cross-entropy loss was employed to quantify the difference between the model’s predictions and true labels. The Stochastic Gradient Descent (SGD) optimizer with a momentum of 0.9 was used to update model weights and biases. The input batch size was set to 32, the initial learning rate to 0.001, and a step-wise learning rate scheduling strategy was adopted with a step size of 20 and gamma of 0.3. All experiments were conducted under identical conditions, with code implemented in PyTorch on the Pycharm platform, and accelerated with an NVIDIA GeForce RTX 3090 GPU.
# Files used in this task  
...    
！[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14062381.svg)

'<FPAN      
     * --checkpoint  
     * --dataset  
     * --log  
     * --my_model  
     * --train>'  

# Get the experimental results by running the train file
