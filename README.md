# Copyright Protection and Accountability of Generative AI Attack, Watermarking, and Attribution
Project source code of Copyright Protection and Accountability of Generative AI Attack, Watermarking, and Attribution
This project mainly contains three parts: RQ1, RQ2, RQ3. Please follow the relative instructions to run experiments for each part. 


## RQ1
### Datasets and Models

**Datasets**

CelebA
```
cd stargan
bash download.sh celeba
```

Horse2zebra
```
cd cyclegan
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

CelebAhq
```
cd starganv2
bash download.sh celeba-hq-dataset
```

**Models**

StarGAN
[StarGAN repository](https://github.com/yunjey/stargan)
```
bash download.sh pretrained-celeba-128x128
```
StarGANv2
[StarGAN_V2 repository](https://github.com/clovaai/stargan-v2)
```
cd starganv2
bash download.sh pretrained-network-celeba-hq
```
CycleGAN
 [CycleGAN official repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
```
cd cyclegan
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
CUT
 [CUT repository](https://github.com/taesungp/contrastive-unpaired-translation)
```
cd CUT
wget http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar
tar -xf pretrained_models.tar
```
AttentionGAN
 [AttentionGAN repository](https://github.com/Ha0Tang/AttentionGAN)
```
cd attGAN
sh ./scripts/download_attentiongan_model.sh horse2zebra
```


### Attack Testing

Here are bash commands for testing vanilla attacks on each different architecture.
```
# StarGAN Attack Test
cd stargan
python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_128/models' --result_dir='stargan_celeba_256/results_test' --test_iters 200000 --batch_size 1

# GANimation Attack Test
cd ganimation
python main.py --mode animation

# starGAN_V2 Attack Test
cd pix2pixHD
python main.py --mode eval_attack --num_domains 2 --w_hpf 1 --resume_iter 100000  --train_img_dir data/celeba_hq/train --val_img_dir data/celeba_hq/val                --checkpoint_dir expr/checkpoints/celeba_hq --eval_dir expr/eval/celeba_hq --val_batch_size 1

# CycleGAN Attack Test
cd cyclegan
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

# CUT Attack Test
cd CUT
python testwithattack.py --dataroot ./datasets/horse2zebra --name horse2zebra_cut_pretrained --CUT_mode CUT --phase test

# AttentionGAN Attack Test
python testwithattack.py --dataroot ./datasets/horse2zebra --name horse2zebra_pretrained --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 100 --epoch latest
```


## RQ2 Trainingset Watermark


## RQ2 GAN-Model Watermark
RQ2 GAN-Model watermark is implemented based on the  [GAN IPR Protection repository](https://github.com/dingsheng-ong/ipr-gan)

### Train
```bash
$ cd RQ2/RQ2_watermark_model
$ python train.py -c configs/<path-to-yaml-file>
```
### Evaluate
```bash
$ cd RQ2/RQ2_watermark_model
$ python eval.py -l log/<directory> -s sample/
```
## RQ3
### Dataset preparation


The scripts expects one directory as input, containing multiple directories each with at least 27,000 images.
These directories will get encoded with labels in the order of appearence, i.e., encoded as follows:

```
data
 |--- A_lsun 	-> label 0
 |--- B_ProGAN 	-> label 1
 	...
```
[data_celebA](www.riri.com) (click to download) includes the generation results for styleganv2, stargan, starganv2, ddgan, and styleswing, and the real images from celebA dataset.
[data_h2z](https://drive.google.com/file/d/1EnVTR2Xmphh5UViTARQKA5NVQf1f9tyd/view?usp=sharing) (click to download) includes the zebra generation results for cyclegan, attentiongan and CUT, and the real zebra images from horse2zebra dataset

```
# CelebA
python prepare_dataset.py /data_celebA -l tfrecords

# Horse2Zebra
python prepare_dataset.py /data_h2z -l tfrecords
```

### Computing Statistics

To compute all of our statistics we utilize the `compute_statistics.py` script. This script is run on the raw (cropped) image files.
```
# CelebA
python compute_statistics.py 5000 data_celebA/CelebaRealImage,REAL  data_celebA/StyleSwing_CelabA,StyleSwin data_celebA/DDGAN_CelabA,DDGAN data_celebA/StarGAN_CelabA,STARGAN data_celebA/StarGAN_v2_CelabA,STARGANV2 data_celebA/StyleGAN_CelabA,STYLEGAN

# Horse2Zebra
python compute_statistics.py 1000 data_h2z/zebra_RealImage,REAL data_h2z/attention_h2z,AttGAN data_h2z/cut_h2z,CUT data_h2z/cyclegan_h2z,CYCLEGAN
```

### Experiments
**Training your own models**

After you have converted the data files as laid out above, you can train a new classifier:
```
# CelebA
python classifier.py train log2 data_color_dct_log_scaled_normalized_train_tf data.tfrecords data_color_dct_log_scaled_normalized_val_tf data.tfrecords -b 32 -e 100 --l2 0.01 --classes 6 --image_size 256

# Horse2Zebra
python classifier.py train log2 data_color_dct_log_scaled_normalized_train_tf/data.tfrecords data_color_dct_log_scaled_normalized_val_tf/data.tfrecords -b 32 -e 100 --l2 0.01 --classes 4 --image_size 256
```

**Testing**
```
python classifier.py test {model_name} data_color_dct_log_scaled_normalized_test_tf/data.tfrecords -b 32 --image_size 256
```

