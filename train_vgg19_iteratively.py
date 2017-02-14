import argparse
import train_vgg19
import tensorflow as tf

EPOCH_TRAINABLE = {2: 'fc6',3: 'conv5_1', 4: 'conv4_1', 5: 'conv3_1', 6:'conv2_1', 7:'conv1_1'}

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=10, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--gray_input", action="store_true", help="Treat input image as grayscale image.")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--crop_size", type=int, default=256, help="size to crop image into.")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--trainable_layer", default="fc6", choices=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1", "fc6"])
parser.add_argument("--prev_trainable_layer", default="fc6", choices=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1", "fc6"])
a = parser.parse_args()

max_epochs = a.max_epochs
for epoch, trainable_layer in EPOCH_TRAINABLE.iteritems():

    # Temporary fix.
    if trainable_layer == 'fc6':
        continue

    if epoch > max_epochs:
        break
    if trainable_layer == "conv1_1":
        a.max_epochs = max_epochs
    else:
        a.max_epochs = epoch
    print("Current max epoch: %d, current trainable layer: %s" %(epoch, trainable_layer))
    a.prev_trainable_layer = a.trainable_layer
    a.trainable_layer = trainable_layer
    train_vgg19.main(a)
    a.checkpoint = a.output_dir
    tf.reset_default_graph() # Without doing this, the tf graph will still exist after the function exists.