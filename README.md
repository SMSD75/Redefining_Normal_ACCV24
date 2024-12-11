# Redefining_Normal_ACCV24

## Evaluation

For evaluation, we refer to the multiobject-bench folder. In this section, we summarize the evaluation process and results. The evaluation involves benchmarking our model against multiple object detection tasks to assess its performance and robustness.

## Training

To start training from scratch, execute dense_tuning.py or leo_tuning.py. The training process uses a Vision Transformer (ViT) backbone with dense tuning capabilities. By default, the configuration is set up for single GPU training with the following parameters:
```python
python dense_tuning.py --device cuda:0 --batch_size 32 --input_size 224 --num_epochs 100 --num_prototypes 20 --dataset pascal --abnormal_class 0
```


### Key Training Parameters

- **Learning rate**: Initial `1e-4`, peak `1e-3`
- **Weight decay**: `1e-2`
- **Input resolution**: `224x224` pixels
- **Batch size**: `32`
- **Number of prototypes**: `20`
- **Training epochs**: `100`

The model performs validation every epoch and logs performance metrics using wandb. For different datasets, the training pipeline automatically adjusts the data transformations:


## Setting Up the Environment

To set up the environment, please use the following repositories:

- [Leopart](https://github.com/MkuuWaUjinga/leopart/tree/main)
- [MaskAlign](https://github.com/OpenDriveLab/maskalign)

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 📣:
``` 
@inproceedings{salehi2024redefining,
  title={Redefining Normal: A Novel Object-Level Approach for Multi-Object Novelty Detection},
  author={Salehi, Mohammadreza and Gavves, Nikolaos Apostolikas Efstratios and Snoek, Cees GM and Asano, Yuki M},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={402--418},
  year={2024}
}

```
