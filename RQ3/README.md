## Research Question 3

### Setup:
The setup of research question 1 is based on the experiments in paper:
<br>Leveraging Frequency Analysis for Deep Fake Image Recognition</br>
**[[Paper]](https://arxiv.org/abs/2003.08685)** **[[Github Repo]](https://github.com/RUB-SysSec/GANDCTAnalysis)**

### Dataset preparation


The scripts expects one directory as input, containing multiple directories each with at least 27,000 images.
These directories will get encoded with labels in the order of appearence, i.e., encoded as follows:

```
data
 |--- A_lsun 	-> label 0
 |--- B_ProGAN 	-> label 1
 	...
```
[data_celebA](https://drive.google.com/file/d/11uu1OdWs-lI1fbBDEkNXh1HCACfwAD-s/view?usp=sharing) (click to download) includes the generation results for styleganv2, stargan, starganv2, ddgan, and styleswing, and the real images from celebA dataset.
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