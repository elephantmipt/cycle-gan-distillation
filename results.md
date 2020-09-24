# Results

### Dataset

For all my experiments I use monet2photo dataset, which consists of Claude Monet painting and landscape images from Flickr. All examples are generated with photo to painting generator.

### Training teacher

The first task was implementing Cycle GAN. I used catalyst and callbacks to separate generator and discriminator training phases.

I trained teacher network for about 60 hours \(62k iterations\) to get relatively good results, here is an output images example:

|  |  |
| :--- | :--- |
| ![](.gitbook/assets/snimok-ekrana-2020-09-24-v-10.59.46.png)  | ![](.gitbook/assets/snimok-ekrana-2020-09-24-v-11.00.11.png)  |

I will tell you more about an artifacts latter.

### Distillation

I tried to use scheme proposed in the previous section. I mentioned that student converges extremely fast \(about three ours and 30k iterations\) to results same as a teacher.

|  |  |
| :--- | :--- |
| ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.50.00.png)   | ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.50.13.png) |

### Results reliableness

After this experiment I thought that this is it, but after checking if my results useful I also trained the student network without layer transfer and initialize other parts of network with random weights. The convergence were slightly worse in comparasion to teacher training process. It takes about two days to reach  But the images were pretty well but with slightly more artifacts.

|  |  |
| :--- | :--- |
| ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.42.24.png)  | ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.43.26.png)  |

### Using pre-trained network

I spend about day to somehow improve teacher quality. I find that in official CycleGAN [repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) there is some pre-trained weights for my dataset. I successfuly tried to match state dict of my model and pre-trained. But get corious results. So after feeding images to network I got this:

|  |  |
| :--- | :--- |
|  ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-21.35.05.png)  | ![](.gitbook/assets/snimok-ekrana-2020-09-23-v-22.29.01%20%281%29.png)  |

Interesting that it is a kinda similar to[ layer visualizations](https://distill.pub/2017/feature-visualization/) in convolution networks.  
I tried to tune it, but even with big weight for identical loss I failed to remove this color inversions.

### Artifacts

I faced with two types of artifact the first is red or green points on the dark parts of images:

![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.55.09.png)

And chessboard artifact on the sky:

![](.gitbook/assets/snimok-ekrana-2020-09-23-v-23.43.26.png)

And if the first artifact is not a problem at all. I just need to train my network for more epochs, the chessboard artifact is annoying and I can't handle it with increasing number of iterations. So I found post [here](https://distill.pub/2016/deconv-checkerboard/) which explains problem and suggests to use upsample with convolution. Now I'm trying to train teacher with this trick.

