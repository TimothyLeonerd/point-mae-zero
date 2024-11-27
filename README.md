# Learning 3D Representations from Procedural 3D Programs

[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://point-mae-zero.cs.virginia.edu/
)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2411.17467)
[![Dataset](https://img.shields.io/badge/Dataset-Download%20Here-blue)](https://huggingface.co/datasets/uva-cv-lab/Point-MAE-Zero)


This repository provides the official implementation for the paper [Learning 3D Representations from Procedural 3D Programs](https://arxiv.org/abs/2411.17467). The paper introduces a self-supervised approach for learning 3D representations using procedural 3D programs. The proposed method achieves comparable performance to learning from ShapeNet on various downstream 3D tasks and significantly outperforms training from scratch.

[Xuweiyi Chen](https://xuweiyichen.github.io/), [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)

If you find this code useful, please consider citing:  
```text
@article{chen2024learning3drepresentationsprocedural,
      title={Learning 3D Representations from Procedural 3D Programs}, 
      author={Xuweiyi Chen and Zezhou Cheng},
      year={2024},
      eprint={2411.17467},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17467}, 
}
```
**:warning: Note:** This is a cleanup version. Further edits and refinements are in progress. This note will be removed once the content has been finalized.

Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```
conda create --prefix <PATH_TO_YOUR_ENV> python=3.10
conda activate <PATH_TO_YOUR_ENV>
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
module load gcc/11.4.0
module load cuda/11.8.0

git clone https://github.com/UVA-Computer-Vision-Lab/point-mae-zero.git
cd point-mae-zero
pip install -r requirements.txt

cd ./extensions/chamfer_dist
python setup.py install --user

cd ../..

cd ./extensions/emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


Datasets
--------

We provide 152,508 3D synthetic shapes [here](https://huggingface.co/datasets/uva-cv-lab/Point-MAE-Zero).

If you want to use this data to pretrain, please edit the path under ``cfg/pretrain_zeroverse.yaml`` after you download this data.

We also provide example pretrain configs for each experiments we performed. If you would like to render your own data, please read `procedural_data_gen/README.md`. We provide additional tools in order to help you divide ``train`` and ``test`` set in ``tools/make_zeroverse_data_faster.py``. You can put ``train.txt`` and ``test.txt`` under ``data/ZeroVerse``. We provide a few examples under ``data/ZeroVerse``. 

Point-MAE-Zero Pre-training
----------------------

To pretrain Point-MAE-Zero on 3D Synthetic Data training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain_zeroverse.yaml --exp_name <output_file_name>
```

For ZeroVerse with different complexty (i.e. 9), you can use 

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain_zeroverse_9.yaml --exp_name <output_file_name>
```

## 5. Point-MAE Fine-tuning

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
Part segmentation on ShapeNetPart, run:
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --root path/to/data --learning_rate 0.0002 --epoch 300
```

Visualization
-------------

Visulization of pre-trained model on either ShapeNet validation set or 3D synthetic Data validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config cfgs/pretrain.yaml --exp_name <name>
```

Acknowledgements
----------------

We would like to acknowledge the following repositories and users for releasing very valuable code and datasets:

- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE?tab=readme-ov-file) for releasing Point-MAE pretrain and evaluation scripts. Large portion of code in this repo are from Point-MAE.
- [Zeroverse](https://github.com/desaixie/zeroverse) for releasing 3D object procedural generated script using Blender
