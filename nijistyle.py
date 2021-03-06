#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from general_util import *
from nijistyle_util import stylize

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
SEMANTIC_MASKS_WEIGHT = 1.0
SEMANTIC_MASKS_NUM_LAYERS = 4
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
PRINT_ITERATIONS = 100
CHECKPOINT_ITERATIONS = 100
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', type=str,
                        dest='content', help='content image. If left blank, it will switch to texture generation mode.',
                        metavar='CONTENT', default='', required=False)
    parser.add_argument('--save_dir', type=str,
                        dest='save_dir', help='save_dir.',
                        metavar='SAVE_DIR', required=False)
    parser.add_argument('--styles',
                        dest='styles', nargs='+', help='one or more style images',
                        metavar='STYLE', required=True)
    parser.add_argument('--use_semantic_masks',
                        dest='use_semantic_masks', help='If true, it accepts some additional image inputs. They '
                                                        'represent the semantic masks of the content and style images.'
                                                        '(default %(default)s).', action='store_true')
    parser.set_defaults(use_semantic_masks=False)
    parser.add_argument('--semantic_masks_weight',
                        dest='semantic_masks_weight', help='The weight given to semantic masks with respect to other '
                                                           'features generated by the vgg network.',
                        metavar='SEMANTIC_MASKS_WEIGHT', required=False, type=float, default=SEMANTIC_MASKS_WEIGHT)
    parser.add_argument('--output_semantic_mask',
                        dest='output_semantic_mask', help='one content image semantic mask',
                        metavar='OUTPUT_SEMANTIC_MASK', required=False)
    parser.add_argument('--style_semantic_masks',
                        dest='style_semantic_masks', nargs='+', help='one or more style image semantic masks',
                        metavar='STYLE_SEMANTIC_MASKS', required=False)
    parser.add_argument('--semantic_masks_num_layers', type=int, dest='semantic_masks_num_layers',
                        help='number of semantic masks per content or style image (default %(default)s).',
                        metavar='SEMANTIC_MASKS_NUM_LAYERS', default=SEMANTIC_MASKS_NUM_LAYERS)
    parser.add_argument('--content_img_style_weight_mask', dest='content_img_style_weight_mask',
                        help='The path to one black-and-white mask specifying how much we should "stylize" each pixel '
                             'in the outputted image. The areas where the mask has higher value would be stylized more '
                             'than other areas. A completely white mask would mean that we stylize the output image '
                             'just as before, while a completely dark mask would mean that we do not stylize the '
                             'output image at all, so it should look pretty much the same as content image. If you do '
                             'not wish to use this feature, just leave it blank (default %(default)s).',
                        metavar='CONTENT_IMG_STYLE_WEIGHT_MASK', default='', required=False)
    parser.add_argument('--output',
                        dest='output', help='Output path. (default %(default)s).',
                        metavar='OUTPUT', default='output/default.jpg', required=False)
    parser.add_argument('--checkpoint-output',
                        dest='checkpoint_output', help='Formatted string for checkpoint output. This string should '
                                                       'contain at least one %s. (default %(default)s).',
                        metavar='OUTPUT_CHECKPOINT', default='output_checkpoint/default_%s.jpg', required=False)
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--width', type=int,
                        dest='width', help='Input and output height. All content images and style images should be '
                                           'automatically scaled accordingly. (default %(default)s).',
                        metavar='WIDTH', default=256, required=False)
    parser.add_argument('--height', type=int, dest='height',
                        help='Input and output height. All content images and style images should be automatically '
                             'scaled accordingly. (default %(default)s).',
                        metavar='HEIGHT', default=256, required=False)
    parser.add_argument('--use_mrf',
                        dest='use_mrf', help='If true, it uses Markov Random Fields loss instead of Gramian loss. '
                                             '(default %(default)s).', action='store_true')
    parser.set_defaults(use_mrf=False)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='How much we weigh the content loss (default %(default)s).',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='How much we weigh the style loss (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-blend-weights', type=float,
                        dest='style_blend_weights', help='If given multiple styles as input, this determines how much '
                                                         'it weighs each style.',
                        nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate', help='Learning rate (default %(default)s).',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--initial',
                        dest='initial', help='The initial image that the program starts with. If left blank, it will '
                                             'start with random noise.',
                        metavar='INITIAL', required=False)
    parser.add_argument('--print-iterations', type=int,
                        dest='print_iterations', help='The program prints the current losses every this number of '
                                                      'rounds.',
                        metavar='PRINT_ITERATIONS', default=PRINT_ITERATIONS, required=False)
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='The program saves the current image every this number of '
                                                           'rounds.',
                        metavar='CHECKPOINT_ITERATIONS', default=CHECKPOINT_ITERATIONS, required=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    content_image = None
    if options.content != '':
        print('reading content image %s' % options.content)
        content_image = read_and_resize_images(options.content, options.height, options.width)
    style_images = read_and_resize_images(options.styles, None, None)  # We don't need to resize style images.

    target_shape = (1, int(options.height), int(options.width), 3)

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight
                               for weight in style_blend_weights]

    output_semantic_mask = None
    style_semantic_masks = None

    initial = options.initial
    if initial is not None:
        raise NotImplementedError
        # initial = imread(initial, shape=(options.height, options.width))

    content_img_style_weight_mask = None

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    checkpoint_dir = os.path.dirname(options.checkpoint_output)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    output_dir = os.path.dirname(options.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if options.save_dir is None:
        print("No savedir input, using standard pretrained vgg19 net instead.")

    for iteration, image in stylize(network=None, content=content_image, styles=style_images,
                                    shape=target_shape, iterations=options.iterations,save_dir=options.save_dir,
                                    content_weight=options.content_weight, style_weight=options.style_weight,
                                    tv_weight=options.tv_weight, style_blend_weights=style_blend_weights,
                                    learning_rate=options.learning_rate, initial=initial, use_mrf=options.use_mrf,
                                    use_semantic_masks=options.use_semantic_masks,
                                    output_semantic_mask=output_semantic_mask,
                                    style_semantic_masks=style_semantic_masks,
                                    semantic_masks_weight=options.semantic_masks_weight,
                                    print_iterations=options.print_iterations,
                                    checkpoint_iterations=options.checkpoint_iterations,
                                    semantic_masks_num_layers=options.semantic_masks_num_layers,
                                    content_img_style_weight_mask=content_img_style_weight_mask):
        output_file = None
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = str(options.output)
        if output_file:
            # imsave(output_file, image)
            with open(output_file,'w') as f:
                f.write(image)


if __name__ == '__main__':
    main()

"""
python nijistyle.py --content=1.png --styles=4.png --save_dir=UECFOOD256_train_iter/ --output=output/test.jpg --iterations=1000 --checkpoint-output="output_checkpoint/test_%s.jpg" --checkpoint-iterations=10 --width=64 --height=64 --print-iterations=10 --learning-rate=10 --style-weight=100 --content-weight=0 --tv-weight=0


python nijistyle.py --content=1.png --styles=4.png --save_dir=UECFOOD256_train_iter/ --output=output/test.jpg --iterations=5000 --checkpoint-output="output_checkpoint/test_%s.jpg" --checkpoint-iterations=10 --width=256 --height=256 --print-iterations=10 --learning-rate=0.1 --style-weight=100 --content-weight=5 --tv-weight=5
python nijistyle.py --content=1.png --styles=4.png --output=output/native_vgg19_test.png --iterations=5000 --checkpoint-output="output_checkpoint/native_vgg19_test_%s.png" --checkpoint-iterations=10 --width=256 --height=256 --print-iterations=10 --learning-rate=0.1 --style-weight=100 --content-weight=5 --tv-weight=5
python nijistyle.py --content=1.png --styles=feed_forward_style_images/4.jpg --output=output/native_vgg19_test.png --iterations=5000 --checkpoint-output="output_checkpoint/native_vgg19_test_%s.png" --checkpoint-iterations=10 --width=256 --height=256 --print-iterations=10 --learning-rate=0.1 --style-weight=100 --content-weight=5 --tv-weight=5

python nijistyle.py --content=feed_forward_style_images/1.jpg --styles=feed_forward_style_images/4.jpg --save_dir=UECFOOD256_train_iter/ --output=output/test.jpg --iterations=1000 --checkpoint-output="output_checkpoint/test_%s.jpg" --checkpoint-iterations=10 --width=256 --height=256 --print-iterations=10 --learning-rate=0.05 --style-weight=100 --content-weight=5 --tv-weight=5
python nijistyle.py --content=feed_forward_style_images/1.jpg --styles=feed_forward_style_images/4.jpg --output=output/native_vgg19_test.jpg --iterations=1000 --checkpoint-output="output_checkpoint/native_vgg19_test_%s.jpg" --checkpoint-iterations=10 --width=256 --height=256 --print-iterations=10 --learning-rate=0.05 --style-weight=100 --content-weight=5 --tv-weight=5
"""
