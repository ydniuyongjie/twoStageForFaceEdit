"""
Learns a matrix of Z-Space directions using a pre-trained BigGAN Generator.
Modified from train.py in the PyTorch BigGAN repo.
"""
import os
import os.path as osp
import sys
import time
import json
import  numpy as np
from lib import *
import torch
import torch.nn as nn
import torch.optim
import argparse
from models.gan_load import build_biggan, build_stylegan2_hessian
import shutil
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from lib.aux import sample_z, TrainingStatTracker, update_progress, update_stdout, sec2dhms
from hessian_penalty import hessian_penalty
from orojar import orojar
from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT
from latent_deformator import Deformator


def log_progress(stat_tracker,stats_json,iteration, mean_iter_time, elapsed_time, eta,args):
        """Log progress in terms of batch accuracy, classification and regression losses and ETA.

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion

        """
        # Get current training stats (for the previous `self.params.log_freq` steps) and flush them
        stats = stat_tracker.get_means()

        # Update training statistics json file
        with open(stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(stats_json, 'w') as out:
            json.dump(stats_dict, out)

        # Flush training statistics tracker
        stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            args.batch_size, iteration, args.max_iter),args.max_iter, iteration + 1)
        if iteration < args.max_iter - 1:
            print()
        print("      \\__penalty loss          : {:.08f}".format(stats['total_loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time      : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time        : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                 : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(7)

def get_starting_iteration(checkpoint_file,support_sets, reconstructor=None):
    """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
    iteration; also load checkpoint weights to `support_sets` and `reconstructor`. Otherwise, set starting
    iteration to 1 in order to train from scratch.

    Returns:
        starting_iter (int): starting iteration

    """
    starting_iter = 1
    if osp.isfile(checkpoint_file):
        checkpoint_dict = torch.load(checkpoint_file)
        starting_iter = checkpoint_dict['iter']
        support_sets.load_state_dict(checkpoint_dict['deformator'])       
    return starting_iter

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def main():
    parser = argparse.ArgumentParser(description="hessian_penalty training script")
    parser.add_argument('--gan-type', type=str, choices=GAN_WEIGHTS.keys(), help='set GAN generator model type')
    parser.add_argument('--z-truncation', type=float, default=0.7,help="set latent code sampling truncation parameter")
    parser.add_argument('--stylegan2-resolution', type=int, default=1024, choices=(256, 1024),help="StyleGAN2 image resolution")
    parser.add_argument('--shift-in-w-space', action='store_true', help="search latent paths in StyleGAN2's W-space")
    # === Deformator (D) ======================================================================== #
    parser.add_argument('--path-type',type=str,default="line",help="define type of direction path(line or no-line)")
    parser.add_argument('--deformator-type', type=str, default='ortho',choices=DEFORMATOR_TYPE_DICT.keys(), help='deformator type')
    parser.add_argument('--directions-count', type=int, default=256,help="set number of directions")
    parser.add_argument('--shift-distribution', choices=SHIFT_DISTRIDUTION_DICT.keys(),default='uniform', help="set distribution of shift")
    parser.add_argument('--min-shift', default=0.5)
    parser.add_argument('--shift-scale', default=6.0, help='scale of shift')
    parser.add_argument('--deformator-lr', type=float, default=0.01, help="set learning rate")
    parser.add_argument('--deformator-random-init', type=bool, default=True)
    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=50000, help="set maximum number of training iterations")
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
    parser.set_defaults(tensorboard=True)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()
    # Create output dir and save current arguments
    ##############################################################
    hessian_result_root = "experiments"
    os.makedirs(hessian_result_root, exist_ok=True) 

    wip_dir = osp.join(hessian_result_root,"wip")
    os.makedirs(wip_dir, exist_ok=True)

    complete_dir = osp.join(hessian_result_root, "complete")
    os.makedirs(complete_dir, exist_ok=True)

# -- models directory (support sets and reconstructor, final or checkpoint files)
    pre_models_dir =osp.join(hessian_result_root,"premodels","line")
    if not osp.isdir(pre_models_dir):
        raise NotADirectoryError("Invalid pre_models directory: {}".format(models_dir))

    # ---- Get all files of models directory
    pre_models_dir_files = [f for f in os.listdir(pre_models_dir) if osp.isfile(osp.join(pre_models_dir, f))]

    # ---- Check for support sets file (final or checkpoint)
    pre_deformator_model = osp.join(pre_models_dir, 'deformator.pt')
    if not osp.isfile(pre_deformator_model):
        deformator_checkpoint_files = []
        for f in pre_models_dir_files:
            if 'deformator-' in f:
                deformator_checkpoint_files.append(f)
        deformator_checkpoint_files.sort()
        pre_deformator_model = osp.join(pre_models_dir, deformator_checkpoint_files[-1])  


    experiment_name="Line_2_Hessian_{}_{}".format(args.gan_type,args.batch_size)
    experiment_dir=osp.join(wip_dir,experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    # Create output directory (wip)
  
    # Save args namespace object in json format
    with open(osp.join(experiment_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # Save the given command in a bash script file
    with open(osp.join(experiment_dir, 'command.sh'), 'w') as command_file:
        command_file.write('#!/usr/bin/bash\n')
        command_file.write(' '.join(sys.argv) + '\n')

    models_dir = osp.join(experiment_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    #Set directory for completed experiment
   
    tb_dir = osp.join(experiment_dir, 'tensorboard')
    os.makedirs(tb_dir, exist_ok=True)
    checkpoint_file = osp.join(models_dir, 'checkpoint.pt')
    ################################################################
    stat_tracker = TrainingStatTracker()
    iter_times = np.array([])
    stats_json = osp.join(experiment_dir, 'stats.json')
    if not osp.isfile(stats_json):
            with open(stats_json, 'w') as out:
                json.dump({}, out)
    ####################################################################
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tb_dir])
      
    tb_url = tb.launch()
    print("#. Start TensorBoard at {}".format(tb_url))
    tb_writer = SummaryWriter(log_dir=tb_dir)
    #########################################################################

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
    # === StyleGAN ===
    elif args.gan_type == 'StyleGAN2':
        G = build_stylegan2_hessian(pretrained_gan_weights=GAN_WEIGHTS[args.gan_type]['weights'][args.stylegan2_resolution],
                            resolution=args.stylegan2_resolution,
                            shift_in_w_space=args.shift_in_w_space)
    else:
        raise NotImplementedError
    
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

    
    # Load pre-trained weights and set to evaluation mode
    print("  \\__Pre-trained weights: {}".format(pre_deformator_model))
    D.load_state_dict(torch.load(pre_deformator_model, map_location=lambda storage, loc: storage))


    # Set `generator` to evaluation mode, `deformator` to training mode, and upload
    # models to GPU if `self.use_cuda` is set (i.e., if args.cuda and torch.cuda.is_available is True).
    #                   direction_indicators = torch.eye(args.directions_count)
    if use_cuda:
        G.cuda().eval()
        D.cuda().train()       
    else:
        G.eval()
        D.train()
    
    
    # Set deformator optimizer
    deformator_optim = torch.optim.Adam(D.parameters(), lr=args.deformator_lr)
    # Get starting iteration断点训练
    checkpoint_file=osp.join(models_dir,"checkpoint.pt")
    starting_iter = get_starting_iteration(checkpoint_file,D)
    # Check starting iteration
    if starting_iter == args.max_iter:
        complete_dir=osp.join(complete_dir,experiment_name)
        print("#. This experiment has already been completed and can be found @ {}".format(experiment_dir))
        print("#. Copy {} to {}...".format(experiment_dir, complete_dir))
        try:
            shutil.copytree(src=experiment_dir, dst=complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'),dirs_exist_ok="true")
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))
        sys.exit()

    print("#. Start training from iteration {}".format(starting_iter))
    all_time_start = time.time()
    # Start training
    for iteration in range(starting_iter, args.max_iter + 1):

        # Get current iteration's start time
        iter_time_start = time.time()

        # Set gradients to zero
        G.zero_grad()
        D.zero_grad()
       

        # Sample latent codes from standard (truncated) Gaussian -- torch.Size([batch_size, generator.dim_z])
        z = sample_z(batch_size=args.batch_size, dim_z=G.dim_z, truncation=args.z_truncation)
        if use_cuda:
            z = z.cuda()

        target_indices = torch.randint( 0, args.directions_count, [args.batch_size], device='cuda')
        shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = args.shift_scale * shifts #self.p.shift_scale is Coefficient(系数)
        shifts[(shifts < args.min_shift) & (shifts > 0)] = args.min_shift
        shifts[(shifts > -args.min_shift) & (shifts < 0)] = -args.min_shift
        latent_dim=D.input_dim

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        col_sample = torch.zeros([args.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            col_sample[i][index] += val
        

        # Calculate shift vectors for the given latent codes -- in the case of StyleGAN2, if --shift-in-w-space is
        # set, the calculate of the shifts will be done in the W-space
        # shift = target_shift_magnitudes.reshape(-1, 1) * S(support_sets_mask, G.get_w(z) if args.shift_in_w_space else z)
        
        # penalty = orojar(G, z, path_idx=col_sample, G_z=None, path_matrix=D, multiple_layers=False,dir_count=args.directions_count).mean()      
        penalty = hessian_penalty(G, z, path_idx=col_sample, G_z=None, path_matrix=D, multiple_layers=False,dir_count=args.directions_count).mean()      

        penalty.backward()

        # Perform optimization step (parameter update)
        deformator_optim.step()

        # Update statistics tracker
        stat_tracker.update(accuracy=None,classification_loss=None,regression_loss=None,total_loss=penalty.item())
        # Update tensorboard plots for training statistics
        if args.tensorboard:
            for key, value in stat_tracker.get_means().items():
                tb_writer.add_scalar(key, value, iteration)       

        # Get time of completion of current iteration
        iter_time_end = time.time()

        # Compute elapsed time for current iteration and append to `iter_times`
        iter_times = np.append(iter_times, iter_time_end - iter_time_start)

        # Compute elapsed time so far
        elapsed_time = iter_time_end - all_time_start

        # Compute rolling mean iteration time
        mean_iter_time = iter_times.mean()

        # Compute estimated time of experiment completion估算总体时间
        eta = elapsed_time * ((args.max_iter - iteration) / (iteration - starting_iter + 1))

        # Log progress in stdout
        if iteration % args.log_freq == 0:
            log_progress(stat_tracker,stats_json,iteration, mean_iter_time, elapsed_time, eta,args)

        # Save checkpoint model file and support_sets model state dicts after current iteration
        if iteration % args.ckp_freq == 0:
            # Build checkpoint dict
            checkpoint_dict = {
                'iter': iteration,
                'deformator': D.state_dict()
            }
            torch.save(checkpoint_dict, checkpoint_file)
    # === End of training loop ===

    # Get experiment's total elapsed time
    elapsed_time = time.time() - all_time_start

    # Save final support sets model
    deformator_model_filename = osp.join(models_dir, 'deformator.pt')
    torch.save(D.state_dict(), deformator_model_filename)
  

            
    #############################################################################
    #                    train finished                                         #
    #                    copy result                                            #
    #############################################################################
    for _ in range(10):
            print()
    print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

    print("#. Copy {} to {}...".format(wip_dir, complete_dir))
    try:
        shutil.copytree(src=wip_dir, dst=complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'),dirs_exist_ok="true")
        print("  \\__Done!")
    except IOError as e:
        print("  \\__Already exists -- {}".format(e))


if __name__ == '__main__':
    main()
