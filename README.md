# 3Dhumanheads
Docker Setup for Blender : 
    - bpy blender file to render out images from OBJ with background and material 
    - Docker file to run bpy files python 3.7 among other things needed 
bachelor_welker.pdf : PDF to bachelor theisis of Bjoern Bastain Welker 
net.py : implemenation of the networks described in bachelor_welker.pdf
losses.py : implemenation of the losses decribed in bachelor_welker.pdf
reqirements.txt : some python reqirements 
If needed a python conda envorment can be included to run erverything 
Torch Sphere Tracer : 
    - adaption of a "normal" CUDA sphere tracer to also accept SDFs with color componenets 
    - erverything still runs on the GPU 
    - might need a sprecial version of CUDA for the istallation of pytorch3d (see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)