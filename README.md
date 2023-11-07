This is the official repository of 

**MetaTransformer: A Unified Framework for Transformer-based Video Anomaly Detection.**

## Setup
```bash
conda create -n meta_transformer python=3.10
conda activate meta_transformer
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset Preparation
Download the [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) dataset and structure the data as follows:
```
data/dota/
  annotations
    .json
  frames
    0qfbmt4G8Rw_000072
      images
        .jpg
      times.txt
    0qfbmt4G8Rw_000306
      images
        .jpg
      times.txt
    ...
  metadata
    metadata_train.json
    metadata_val.json
    train_split.txt
    val_split.txt
```

## Usage
To use our model, follow the code snippet below:
```bash
cd Transformer_based

# Train, Test, Demo and Play C3D + LSTM
bash scripts/train_c3d_lstm.sh
bash scripts/eval_c3d_lstm.sh
bash scripts/demo_c3d_lstm.sh
bash scripts/play_c3d_lstm.sh

# Train, Test and Demo Swin-b + LSTM
bash scripts/train_swinb_lstm.sh
bash scripts/eval_swinb_lstm.sh
bash scripts/demo_swinb_lstm.sh
bash scripts/play_swinb_lstm.sh

# Train, Test and Demo Swin-t + LSTM
bash scripts/train_swint_lstm.sh
bash scripts/eval_swint_lstm.sh
bash scripts/demo_swint_lstm.sh
bash scripts/play_swint_lstm.sh
```

## MetaAnomaly Model Zoo
TBA.

## Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {MetaTransformer: A Unified Framework for Transformer-based Video Anomaly Detection},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/IceIce1ce/MetaTransformer},
  year         = {2023}
}
```

## Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

##  Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [IMPLabUniPr/movad](https://github.com/IMPLabUniPr/movad)
* [haofanwang/video-swin-transformer-pytorch](https://github.com/haofanwang/video-swin-transformer-pytorch)
<!--te-->
