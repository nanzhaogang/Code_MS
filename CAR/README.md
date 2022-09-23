
# Contents

- [Contents](#contents)
- [Learned Image Downscaling for Upscaling Using Content Adaptive Resampler (CAR)](#learned-image-downscaling-for-upscaling-using-content-adaptive-resampler-car)
- [Pretrained model](#pretrained-model)
- [Training Parameter description](#training-parameter-description)
    - [Example](#example)
        - [Dataset](#dataset)
        - [Train Model](#train-model)
        - [Evaluate Model](#evaluate-model)
        - [Infer](#infer)
        - [Result](#result)

# Learned Image Downscaling for Upscaling Using Content Adaptive Resampler (CAR)

CAR is an efficient image downscaling and upscaling method to make resources saving by only generating, storing and transmitting a single downscaled version for preview and upscaling it to high resolution when details are going to be viewed.  It employed a SR model to try the best to recover HR images while adaptively adjusting the downscaling model to produce LR images.

# Pretrained model

Quantitative evaluate result (PSNR / SSIM) of different image downscaling method for SR on benchmar datasets: SET5, SET14, BSD100, URBAN100 AND DIV2K (VALIDATION SET).  Model for MindSpore:

<table>
   <tr>
      <td>scale </td>
      <td>model</td>
      <td>Set5</td>
      <td>Set14</td>
       <td>B100</td>
    <td>Urban100</td>
           <td>DIV2K(Val)</td>
      <td>ckpt</td>
   </tr>
   <tr>
      <td rowspan="2">4x</td>
      <td>kernel generator net</td>
      <td rowspan="2">34.17 / 0.9196</td>
      <td rowspan="2"> 29.49 / 0.8092</td>
      <td rowspan="2"> 30.61 / 0.8427</td>
       <td rowspan="2"> 29.31 / 0.8704</td>
       <td rowspan="2"> 32.68 / 0.8871</td>
       <td><a href="https://download.mindspore.cn/vision/car/2x/kgn.ckpt">ckpt</a></td>
   </tr>
   <tr>
      <td>super resolution net</td>
      <td><a href="https://download.mindspore.cn/vision/car/2x/usn.ckpt">ckpt</a></td>
   </tr>
   <tr>
      <td rowspan="2">2x</td>
      <td>kernel generator net</td>
      <td rowspan="2"> 38.96 / 0.9643</td>
      <td rowspan="2"> 35.84 / 0.9394</td>
      <td rowspan="2"> 33.88 / 0.9221</td>
       <td rowspan="2"> 35.36 / 0.9556</td>
       <td rowspan="2"> 37.92 / 0.9583</td>
       <td><a href="https://download.mindspore.cn/vision/car/4x/kgn.ckpt">ckpt</a></td>
   </tr>
   <tr>
      <td>super resolution net</td>
      <td><a href="https://download.mindspore.cn/vision/car/4x/usn.ckpt">ckpt</a></td>
   </tr>
</table>

# Training Parameter description

| Parameter        | Default      | Description                |
| ---------------- | ------------ | -------------------------- |
| workers          | 1            | Number of parallel workers |
| device_target    | GPU          | Device type                |
| base_lr          | 0.0001       | Base learning rate         |
| end_epoch        | 500          | Number of epoch            |
| scale            | 4          | Downscaling rate           |
| target_dataset   | DIV2KHR      | Dataset name               |
| train_batchsize  | 16           | Number of batch size       |
| checkpoint_path  | ./checkpoint | Path to save checkpoint    |
| image_path       | ./datasets   | Path of training file      |
| eval_proid       | 1            | The period for evaluating  |
| train_repeat_num | 1            | Repeated the training set  |

## Example

Here, how to use CAR model will be introduce as following.

### Dataset

In this work, it employed the DIV2K image dataset for training. So first, you should download the dataset from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/),. Since it focus on how to
downscale images without any supervision, only HR images of the mentioned datasets were utilized.

After you get the dataset, make sure your path is as following:

```text

.datasets/
    └── DIV2K
            ├── DIV2K_train_HR
            |    ├── 0001.png
            |    ├── 0002.png
            |    ├── ...
            ├── DIV2K_valid_HR
            |    ├── 000801.png
            |    ├── 000802.png
            |    ├── ...

```

### Train Model

This work provide 2 different downing scaling rate, 2x and 4x. You can easily to select it by setting training parameter 'scale'. Note that for 4x downscale and the HR image is cropped to 192x192 while for 2x it cropped to 96x96

#### For 2x down scaling

```shell
python train.py --image_path=./database/DIV2K --scale=2 --train_resize=96
```

output:

```text
Epoch:[  0/500], step:[    4/   50], loss:[0.214/0.239], time:305.804 ms, lr:0.00010
Epoch:[  0/500], step:[    5/   50], loss:[0.209/0.233], time:305.325 ms, lr:0.00010
Epoch:[  0/500], step:[    6/   50], loss:[0.184/0.225], time:305.311 ms, lr:0.00010
......
Epoch:[  0/500], step:[   49/   50], loss:[0.050/0.104], time:305.329 ms, lr:0.00010
Epoch:[  0/500], step:[   50/   50], loss:[0.070/0.104], time:306.509 ms, lr:0.00010
Epoch time: 75236.870 ms, per step time: 1504.737 ms, avg loss: 0.104
Validating...
epoce 1, Save model..., m_psnr for 10 images: 23.671447448609797
Validating Done.
```

#### For 4x down scaling

```shell
python train.py --image_path=./database/DIV2K --scale=4 --train_resize=192
```

output

```text
Finish loading train dataset, data_size:50
Checkpoint files will save to /home/car/work/course/application_example/CAR/src/checkpoint/4x
Epoch:[  0/500], step:[    1/   50], loss:[0.227/0.227], time:12271.484 ms, lr:0.00010
Epoch:[  0/500], step:[    2/   50], loss:[0.268/0.248], time:438.299 ms, lr:0.00010
Epoch:[  0/500], step:[    3/   50], loss:[0.290/0.262], time:437.805 ms, lr:0.00010
......
Epoch:[  0/500], step:[   49/   50], loss:[0.080/0.126], time:445.816 ms, lr:0.00010
Epoch:[  0/500], step:[   50/   50], loss:[0.076/0.125], time:444.366 ms, lr:0.00010
Epoch time: 75821.120 ms, per step time: 1516.422 ms, avg loss: 0.125
Validating...
epoce 1, Save model..., m_psnr for 10 images: 22.130077442719646
```

### Evaluate Model

In the testing, four standard datasets, i.e., the Set5, Set14, BSD100 and Urban100 were used as suggested by the EDSR paper.  You can download from [Set5](https://gitee.com/link?target=https%3A%2F%2Fdeepai.org%2Fdataset%2Fset5-super-resolution). After unzip the downloaded file, the struct of directory should as follow:

```text

└── datasets
       ├── Set5
       |    ├── baby.png
       |    ├── bird.png
       |    ├── ...
       ├── Set14
       |    ├── baboon.png
       |    ├── barbara.png
       |    ├── ...
       ├── BSDS100
       |    ├── 101085.png
       |    ├── 101087.png
       |    ├── ...
       ├── Urban100
       |    ├── img_001.png
       |    ├── img_002.png
       |    ├── ...
       └── DIV2K
           ├── DIV2K_train_HR
           |    ├── 0001.png
           |    ├── 0002.png
           |    ├── ...
           ├── DIV2K_valid_HR
           |    ├── 000801.png
           |    ├── 000802.png
           |    ├── ...

```

Then you can execute command as follow:

#### For 2x down scaling

```shell
python eval.py --img_dir=/home/car/datasets/ --scale=2  --target_dataset=Set5
```

#### For 4x down scaling

python eval.py --img_dir=/home/car/datasets/ --scale=4  --target_dataset=Set5

output:

```text
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.27s/it]
Mean PSNR: 38.96
Mean SSIM: 0.9643
```

### Infer

At last, you can use your own image to test your model. Put your image in the img_dir folder, then run eval.py to do inference.

### Result

|<center> Original image        |<center> Downscaling image      |<center> Reconstuct image                |
| ---------------- | ------------ | -------------------------- |
| <center>![show_images](images/orig_img.png)| <center> ![show_images](images/downscale_img.png)|![show_images](images/recon_img.png) |
