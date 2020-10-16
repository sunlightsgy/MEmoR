# MEmoR

This is the official pytorch implementation for the paper ["MEmoR: A Dataset for Multimodal Emotion Reasoning in Videos"](some link) in ACM Multimedia 2020."

## Installation
- Python 3.6
- Clone this repo and install the python dependencies:
```
git clone https://github.com/sunlightsgy/MEmoR.git
cd MEmoR
pip install -r requirements.txt
```

## Datasets
The MEmoR datasets are released on [onedrive](https://tsinghuaeducn-my.sharepoint.com/:f:/g/personal/xin_wang_tsinghua_edu_cn/Eq4cmdFBF1RDjyM_Pd3VRQcBZAr4IZCeYRZ_Dzh_1tFFHg?e=I4DmiC). You should download the License Agreement in this repo and send back to thusgy2012 at gmail.com. Then you will get the password. Once downloaded, please set a soft link to the MEmoR dataset:

```
ln -s /path/to/MEmoR data
```

## Usage
The training and testing configures are set in `train.json` and `test.json`. To switch between the primary and fine-grained emotions, modified `emo_type` in these two files.

### Training
```
python train.py -c train.json -d [gpu_id]
```
### Testing
```
python test.py -c test.json -d [gpu_id] -r /path/to/model
```

## The Pretrain Model
We provide a pretrained model for primary and fine-grained emotions in the data/pretrained on the downloaded datasets.


## Citation

If you use this code or dataset for your research, please cite our papers.

```
@inproceedings{shen2020memor,
  title={MEmoR: A Dataset for Multimodal Emotion Reasoning in Videos},
  author={Shen, Guangyao and Wang, Xin and Duan, Xuguang and Li, Hongzhi and Zhu, Wenwu},
  booktitle={Proceedings of the 28th ACM international conference on Multimedia},
  pages={493--502},
  year={2020},
  organization={ACM}
}
```

## Acknowledgments

This project template is borrowed from the project [PyTorch Template Project](https://github.com/victoresque/pytorch-template). 
