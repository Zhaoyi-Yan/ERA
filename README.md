# News
This paper is accepted by TNNLS (IEEE Transactions on Neural Networks and Learning Systems) 2025.


# Prepare the dataset
When you prepare the imagenet dataset, the folder tree should be:
-- Imagenet
  - train
  - val
  - meta_data

You can directly copy `meta_data` to the corresponding folder.
As the limitation of supplementary size must be lower than 100M, so we delete `train.txt`, which has no harm for evaluation.

Always, you can generate the meta data youself in `add_folder_cls.py`.
For more inforamtion, you can get the details in the function `build_dataloader` in `lib/dataset/builder.py`.

# Replace the Torchvision Resnet with custom resnets.
You can refer to the `tv_resnet_modified.py`, and change your `resnet.py` in the torchvision package in your environment.

It mainly adds `in_c_tec` parameter to the ResNet function, so that we can add `up_fc` layer for adapter ($P$).

# Test
Change `IMAGENET_PATH` in `test_res34_res18.sh` as the imagenet folder path.
If you use anaconda, you should change `YOUR_ANACONDA` to the correct path.
Remember, if using anaconda, you should change the `resnet.py` in the path like `/userhome/anonymous/anaconda/lib/python3.9/site-packages/torchvision/models/resnet.py`.

### Mention
Our code are based on pytorch 1.12, and also has been validated for pytorch 2.0.
Always, the only tiny differences exist the `resnet.py` in torchvision. You can always modify the code yourself, which should be quite easy.

And remember to do the previous two things before running the scripts.

# Train a model 
```bash
bash imagenet/res34_res18.sh
```
You can directly change the residual network name, for example from `tv_resnet34` to `tv_resnet101`, to switch between different teacher or student models. The prefix `tv` indicates that the original torchvision ResNet models are being used.

## Projects based on Image Classification SOTA
* [NeurIPS 2022] [DIST](https://github.com/hunto/DIST_KD): Knowledge Distillation from A Stronger Teacher


# Citation
```cite
@article{yan2025expandable,
  title={Expandable Residual Approximation for Knowledge Distillation},
  author={Zhaoyi Yan and Binghui Chen and Yunfan Liu and Qixiang Ye},
  journal={arXiv preprint arXiv:-},
  year={2025}
}
```