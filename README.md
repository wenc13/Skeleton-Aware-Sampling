# Skeleton-aware Sampling

It is the official PyTorch implementation of the CVPR 2023 paper:  
### [Learnable Skeleton-Aware 3D Point Cloud Sampling](https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_Learnable_Skeleton-Aware_3D_Point_Cloud_Sampling_CVPR_2023_paper.pdf)  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Cheng Wen, Baosheng Yu, Dacheng Tao**

## Abstract
Point cloud sampling is crucial for efficient large-scale point cloud analysis, where learning-to-sample methods have recently received increasing attention from the community for jointly training with downstream tasks. However, the above-mentioned task-specific sampling methods usually fail to  explore the geometries of objects in an explicit manner. In this paper, we introduce a new skeleton-aware learning-to-sample method by learning object skeletons as the prior knowledge to preserve the object geometry and topology information during sampling. Specifically, without labor-intensive annotations per object category, we first learn category-agnostic object skeletons via the medial axis transform definition in an unsupervised manner. With object skeleton, we then evaluate the histogram of the local feature size as the prior knowledge to formulate skeleton-aware sampling from a probabilistic perspective. Additionally, the proposed skeleton-aware sampling pipeline with the task network is thus end-to-end trainable by exploring the reparameterization trick. Extensive experiments on three popular downstream tasks, point cloud classification, retrieval, and reconstruction, demonstrate the effectiveness of the proposed method for efficient point cloud analysis. 

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{wen2023learnable,
  title={Learnable Skeleton-Aware 3D Point Cloud Sampling},
  author={Wen, Cheng and Yu, Baosheng and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17671--17681},
  year={2023}
}
```


## Enviroments
To run this code, you need first install these python packages:  
```
python 3.8  
pytorch 1.10  
torchvision, numpy, tqdm, h5py, plyfile, matplotlib, scipy, networkx  
```

## Dataset

## Usage

## License
Our code is released under MIT License (see LICENSE file for details).
