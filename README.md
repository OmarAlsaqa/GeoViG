# GeoViG
## GeoViG: Geometry-Aware Graph Reasoning for Mobile Vision Tasks in Natural and Medical Images

**Abstract**
This repository contains the implementation of **GeoViG**, a lightweight architecture that bridges the gap between efficient grid-based CNNs/ViTs and explicit Geometric Deep Learning. By introducing *SpreadEdgePool* for geometry-aware downsampling and *GraphMRConv* for vectorized message passing, GeoViG achieves superior parameter efficiency and performance on mobile devices.

This repository is built upon the [MobileViG](https://github.com/SLDGroup/MobileViG) repository.

## Pretrained Models

### Image Classification (ImageNet-1K)

| Model | Params (M) | MACs (G) | Top-1 Acc (%) | Checkpoint |
| :--- | :---: | :---: | :---: | :---: |
| **GeoViG-Ti** | 3.5 | 0.9 | 75.2 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/pth/geovig_ti_5e4_8G_300_75_22/checkpoint.pth) |
| **GeoViG-S** | 5.0 | 1.2 | 77.5 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/pth/geovig_s_5e4_8G_300_77_48/checkpoint.pth) |
| **GeoViG-M** | 10.3 | 2.2 | 80.7 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/pth/geovig_m_5e4_8G_300_80_70/checkpoint.pth) |
| **GeoViG-B** | 19.7 | 4.5 | 82.4 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/pth/geovig_b_5e4_8G_300_82_38/checkpoint.pth) |

For CoreML checkpoint, please check this [Link](https://huggingface.co/OmarAlasqa/GeoViG/tree/main/CoreML)
For IPA chechpoints, please check this [Link](https://huggingface.co/OmarAlasqa/GeoViG/tree/main/IPA)

### Object Detection & Instance Segmentation (COCO 2017)

| Backbone | Params (M) | Box AP | Mask AP | Checkpoint |
| :--- | :---: | :---: | :---: | :---: |
| **GeoViG-M** | 10.3 | 40.7 | 37.7 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/coco_det_seg_pth/geovig_m_det_seg/epoch_12.pth) |
| **GeoViG-B** | 19.7 | 42.5 | 38.9 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/coco_det_seg_pth/geovig_b_det_seg/epoch_12.pth) |

### Medical Image Segmentation

**Kvasir-SEG (Polyp Segmentation)**

| Backbone | mAP | Dice | IoU | Hausdorff Dist. | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GeoViG-M** | 0.990 | 0.945 | 0.909 | 12.94 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/medical/kvasir_geovig_m/checkpoint.pth) |

**Data Science Bowl 2018 (Nuclei Segmentation)**

| Backbone | mAP | Dice | IoU | Hausdorff Dist. | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GeoViG-M** | 0.859 | 0.908 | 0.839 | 5.19 | [HF](https://huggingface.co/OmarAlasqa/GeoViG/blob/main/medical/dsb_geovig_m/checkpoint.pth) |


## Directory Structure

```
.
├── detection/          # Object detection & Instance Segmentation (MMDetection configs & backbones)
├── models/             # GeoViG core model implementation (`geovig.py`)
├── util/               # Utility scripts/helpers
├── main.py             # ImageNet Classification training & evaluation script
└── requirements.txt    # Python dependencies
```


## Usage

### Installation Image Classification

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```bash
conda install mpi4py
```
```bash
pip install -r requirements.txt
```

### Image Classification

#### Train image classification:
```bash
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model geovig_model --output_dir geovig_results
```
For example, to train GeoViG-S:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env main.py --data-path /path/to/imagenet --model geovig_s --output_dir geovig_s_results
```

#### Test image classification:
```bash
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model geovig_model --resume pretrained_model --eval
```
For example:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env main.py --data-path /path/to/imagenet --model geovig_s --resume checkpoints/geovig_s.pth --eval
```


### Installation Object Detection and Instance Segmentation
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```bash
pip install timm
```
```bash
pip install submitit
```
```bash
pip install -U openmim
```
```bash
mim install mmcv-full==1.7.2
```
```bash
mim install mmdet==2.28.0
```

### Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of PVT, PoolFormer, and EfficientFormer for fair comparison.

All commands for object detection and instance segmentation should be run from the `detection/` directory.

#### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).

#### ImageNet Pretraining
Place ImageNet-1K pretrained weights in the appropriate folder or update the config file path.

#### Train object detection and instance segmentation:
```bash
python -m torch.distributed.launch --nproc_per_node num_GPUs --nnodes=num_nodes --node_rank 0 main.py configs/mask_rcnn_geovig_model --geovig_model geovig_model --work-dir Output_Directory --launcher pytorch > Output_Directory/log_file.txt 
```
For example, to train using GeoViG-M backbone:
```bash
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 main.py configs/mask_rcnn_geovig_m_fpn_1x_coco.py --geovig_model geovig_m --work-dir detection_results/ --launcher pytorch
```

#### Test object detection and instance segmentation:
```bash
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --node_rank 0 test.py configs/mask_rcnn_geovig_model --checkpoint Pretrained_Model --eval {bbox or segm} --work-dir Output_Directory --launcher pytorch
```
For example:
```bash
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank 0 test.py configs/mask_rcnn_geovig_m_fpn_1x_coco.py --checkpoint detection_results/latest.pth --eval bbox --work-dir detection_results/ --launcher pytorch
```
