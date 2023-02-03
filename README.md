# Copyright_Protection_of_Image_GANs_Attack-Watermarks-Attribution
Source project code of Copyright Protection of Image Generative Adversarial Networks: Attack, Watermarks, and Attribution




## RQ1
### Datasets and Models
**StarGAN Dataset 128x128**
```
cd stargan
bash download.sh celeba
```
**StarGAN Models 128x128**
```
bash download.sh pretrained-celeba-128x128
```

**StarGANv2 Dataset and Models**

Follow instruction in the [CycleGAN official repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for downloading their models and data.

**CycleGAN Dataset and Models**

Follow instruction in the [CycleGAN official repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for downloading their models and data.

**CUT Models**
```

```


**AttentionGAN Model**
Follow instruction in the [AttentionGAN repository](https://github.com/Ha0Tang/AttentionGAN)
```
sh ./scripts/download_attentiongan_model.sh horse2zebra
```



### Attack Testing

Here are bash commands for testing our vanilla attacks on each different architecture.
```
# StarGAN Attack Test
cd stargan
python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='stargan_celeba_256/results_test' --test_iters 200000 --batch_size 1

# GANimation Attack Test
cd ganimation
python main.py --mode animation

# pix2pixHD Attack Test
cd pix2pixHD
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none

# CycleGAN Attack Test
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```

If you want to change the attack method being used, look into the attack.py scripts in each architecture folder and change the number of iterations, attack magnitude and step size. You can also re-run the class transferring and blur evasion experiments on StarGAN by commenting/uncommenting lines 54-61 in stargan/main.py or modifying the stargan/solver.py script to change the attack type.

In order to change attack types for GANimation you can modify lines 386-470 by commenting out the vanilla attack and uncommenting the attack you want to run. 

## GAN Adversarial Training
In order to run G+D adversarial training on StarGAN run:
```
# StarGAN Adversarial Training
python main.py --mode train --dataset CelebA --image_size 256 --c_dim 5 --sample_dir stargan_both/samples --log_dir stargan_both/logs --model_save_dir stargan_both/models --result_dir stargan_both/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```
If you wish to run vanilla training or generator adversarial training, comment/uncomment the appropriate lines (l.44-49) in stargan/main.py

The G+D adversarially trained model we used in the paper can be downloaded [here](https://drive.google.com/open?id=1xMM7q4w3lczO6Iskj8CWwmNWHBer9RBP).

## Image Translation Network Implementations
We use code from [StarGAN](https://github.com/yunjey/stargan), [GANimation](https://github.com/vipermu/ganimation), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [advertorch](https://github.com/BorealisAI/advertorch). These are all great repositories and we encourage you to check them out and cite them in your work.
