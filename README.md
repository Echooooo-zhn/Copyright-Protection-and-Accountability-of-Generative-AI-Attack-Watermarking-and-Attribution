# Copyright_Protection_of_Image_GANs_Attack-Watermarks-Attribution
Source project code of Copyright Protection of Image Generative Adversarial Networks: Attack, Watermarks, and Attribution




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
