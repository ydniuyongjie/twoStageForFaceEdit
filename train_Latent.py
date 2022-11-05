import argparse
from symbol import parameters
import torch
from lib import *
from models.gan_load import build_biggan, build_proggan, build_stylegan2, build_sngan
from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT
from latent_deformator import Deformator



def main():
    """Discovery LatentSpace in W space -- Training script.

    Options:
        ===[ Pre-trained GAN Generator (G) ]============================================================================
        --gan-type                 : set pre-trained GAN type
        --z-truncation             : set latent code sampling truncation parameter. If set, latent codes will be sampled
                                     from a standard Gaussian distribution truncated to the range [-args.z_truncation,
                                     +args.z_truncation]
        --biggan-target-classes    : set list of classes to use for conditional BigGAN (see BIGGAN_CLASSES in
                                     lib/config.py). E.g., --biggan-target-classes 14 239.
        --stylegan2-resolution     : set StyleGAN2 generator output images resolution:  256 or 1024 (default: 1024)
        --shift-in-w-space         : search latent paths in StyleGAN2's W-space (otherwise, look in Z-space)

        ===[ Deformator (D) ]=========================================================================================
        --directions-count        : set number of directions; i.e.number of interpretable paths
        --shift-distribution      : set distribution of shift,default  UNIFORM
        --min-shift               : shift min  0.5
        --shift-scale             : shift scale 6.0
        --deformator-lr            : set learning rate for learning deformator 

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --min-shift-magnitude      : set minimum shift magnitude
        --max-shift-magnitude      : set maximum shift magnitude
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

        ===[ CUDA ]=====================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="LatentSpace line training script")

    # === Pre-trained GAN Generator (G) ============================================================================== #
    parser.add_argument('--gan-type', type=str, choices=GAN_WEIGHTS.keys(), help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, help="set latent code sampling truncation parameter")
    parser.add_argument('--biggan-target-classes', nargs='+', type=int, help="list of classes for conditional BigGAN")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")

    # === Deformator (D) ======================================================================== #
    parser.add_argument('--path-type',type=str,default="line",help="define type of direction path(line or no-line)")
    parser.add_argument('--deformator-type', type=str, default='ortho',choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
    parser.add_argument('--directions-count', type=int, default=256,help="set number of directions")
    parser.add_argument('--shift-distribution', choices=SHIFT_DISTRIDUTION_DICT.keys(),default='uniform', help="set distribution of shift")
    parser.add_argument('--min-shift', default=0.5)
    parser.add_argument('--shift-scale', default=6.0, help='scale of shift')
    parser.add_argument('--deformator-lr', type=float, default=1e-4, help="set learning rate")
    parser.add_argument('--deformator-random-init', type=bool, default=True)

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-type', type=str, choices=RECONSTRUCTOR_TYPES, default='ResNet',help='set reconstructor network type')
    parser.add_argument('--min-shift-magnitude', type=float, default=0.25, help="set minimum shift magnitude")
    parser.add_argument('--max-shift-magnitude', type=float, default=0.45, help="set shifts magnitude scale")
    parser.add_argument('--reconstructor-lr', type=float, default=1e-4,help="set learning rate for reconstructor R optimization")

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=0.25, help="regression loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")

    # === CUDA ======================================================================================================= #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args)

    # CUDA
    use_cuda = False
    multi_gpu = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if torch.cuda.device_count() > 1:
                multi_gpu = True
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    print("#. Build GAN generator model G and load with pre-trained weights...")
    print("  \\__GAN type: {}".format(args.gan_type))
    if args.gan_type == 'StyleGAN2':
        print("  \\__Search for paths in {}-space".format('W' if args.shift_in_w_space else 'Z'))
    if args.z_truncation:
        print("  \\__Input noise truncation: {}".format(args.z_truncation))
    print("  \\__Pre-trained weights: {}".format(
        GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution] if args.gan_type == 'StyleGAN2' else
        GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]]))

    # === BigGAN ===
    if args.gan_type == 'BigGAN':
        G = build_biggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                         target_classes=args.biggan_target_classes)
    # === ProgGAN ===
    elif args.gan_type == 'ProgGAN':
        G = build_proggan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]])
    # === StyleGAN ===
    elif args.gan_type == 'StyleGAN2':
        G = build_stylegan2(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space)
    # === Spectrally Normalised GAN (SNGAN) ===
    else:
        G = build_sngan(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][GAN_RESOLUTIONS[args.gan_type]],
                        gan_type=args.gan_type)

    # Build Deformator Object
    print("#. Build Deformator D...")
    print("  \\__Number of Directions      : {}".format(args.directions_count))
    print("  \\__Type of Deformator        : {}".format(args.deformator_type))
    print("  \\__Latent code dim           : {}".format(G.dim_z))
    print("  \\__Scale of Shift            : {}".format(args.shift_scale))
    print("  \\__Min of Shift              : {}".format(args.min_shift))
    

    D = Deformator(shift_dim=G.dim_z,
                   input_dim=G.dim_z,
                   out_dim=None,
                   type=DEFORMATOR_TYPE_DICT[args.deformator_type],
                   random_init=args.deformator_random_init).cuda()

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in D.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    R = Reconstructor(reconstructor_type=args.reconstructor_type,
                      dim=args.directions_count,
                      channels=1 if args.gan_type == 'SNGAN_MNIST' else 3)

    # Count number of trainable parameters打印可以训练的参数个数
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    trn = Trainer_latent(params=args, exp_dir=exp_dir, use_cuda=use_cuda, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, deformator=D, reconstructor=R)


if __name__ == '__main__':
    main()
