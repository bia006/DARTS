## Overview
![overall_structure](./assets/overview.pdf)

## Dependencies and Installation

1. Create Conda Environment

2. Install Dependencies

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)
- Test Set: [WR-SR Dataset](https://drive.google.com/drive/folders/16UKRu-7jgCYcndOlGYBmo5Pp0_Mq71hP?usp=sharing), [CUFED5 Dataset](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.

## Get Started

### Pretrained Models
Downloading the pretrained models from this [link](https://drive.google.com/file/d/1VWQH3FFhc8ZqDilVfXzlZS99qaF3om0-/view?usp=sharing) and put them under `mmsr/checkpoints folder`.

### Test

We provide quick test code with the pretrained model.

1. Modify the paths to dataset and pretrained model in the following yaml files for configuration.

    ```bash
    ./options/test/test.yml
    ```

2.  Check out the results in `./results`.


### Train

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./mmsr/checkpoints` and `./tb_logger` directory.

1. Modify the paths to dataset in the following yaml files for configuration.
   ```bash
   ./options/train/train.yml
   ```

2. Train the network.
   ```bash
   python mmsr/train.py -opt "options/train/train.yml"
   ```
