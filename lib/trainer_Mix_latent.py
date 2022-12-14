from multiprocessing.util import is_exiting
import sys
import os
import os.path as osp
import json
import torch
from torch import nn, true_divide
import torch.backends.cudnn as cudnn
import numpy as np
import time
import shutil
from enum import Enum
from constants import DEFORMATOR_TYPE_DICT, SHIFT_DISTRIDUTION_DICT, WEIGHTS
from hessian_penalty import hessian_penalty

from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from .aux import sample_z,  update_progress, update_stdout, sec2dhms,TrainingMixStatTracker



class ShiftDistribution(Enum):
    NORMAL = 0,
    UNIFORM = 1,


class DataParallelPassthrough(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super(DataParallelPassthrough, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Trainer_Mix_latent(object):
    def __init__(self, params=None, exp_dir=None, use_cuda=False, multi_gpu=False):
        if params is None:
            raise ValueError("Cannot build a Trainer instance with empty params: params={}".format(params))
        else:
            self.params = params
        self.use_cuda = use_cuda
        self.multi_gpu = multi_gpu

        # Use TensorBoard
        self.tensorboard = self.params.tensorboard

        # Set output directory for current experiment (wip)
        self.wip_dir = osp.join("experiments", "wip", exp_dir)

        # Set directory for completed experiment
        self.complete_dir = osp.join("experiments", "complete", exp_dir)

        # Create log sub-directory and define stat.json file
        self.stats_json = osp.join(self.wip_dir, 'stats.json')
        if not osp.isfile(self.stats_json):
            with open(self.stats_json, 'w') as out:
                json.dump({}, out)

        # Create models sub-directory
        self.models_dir = osp.join(self.wip_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        # Define checkpoint model file
        self.checkpoint = osp.join(self.models_dir, 'checkpoint.pt')

        # Setup TensorBoard
        if self.tensorboard:
            # Create tensorboard sub-directory
            self.tb_dir = osp.join(self.wip_dir, 'tensorboard')
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb = program.TensorBoard()
            self.tb.configure(argv=[None, '--logdir', self.tb_dir])
            self.tb_url = self.tb.launch()
            print("#. Start TensorBoard at {}".format(self.tb_url))
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)

        # Define cross entropy loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Array of iteration times
        self.iter_times = np.array([])

        # Set up training statistics tracker
        self.stat_tracker = TrainingMixStatTracker()


    def make_shifts(self, latent_dim):
        target_indices = torch.randint( 0, self.params.directions_count, [self.params.batch_size], device='cuda')
        # target_indices is number of column of A 
        if SHIFT_DISTRIDUTION_DICT[self.params.shift_distribution].name == ShiftDistribution.NORMAL.name:
            shifts = torch.randn(target_indices.shape, device='cuda')
        elif SHIFT_DISTRIDUTION_DICT[self.params.shift_distribution].name == ShiftDistribution.UNIFORM.name:
            shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

        shifts = self.params.shift_scale * shifts #self.p.shift_scale is Coefficient(??????)
        shifts[(shifts < self.params.min_shift) & (shifts > 0)] = self.params.min_shift
        shifts[(shifts > -self.params.min_shift) & (shifts < 0)] = -self.params.min_shift

        try:
            latent_dim[0]
            latent_dim = list(latent_dim)
        except Exception:
            latent_dim = [latent_dim]

        z_shift = torch.zeros([self.params.batch_size] + latent_dim, device='cuda')
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift[i][index] += val

        return target_indices, shifts, z_shift

    def get_starting_iteration(self, deformator, reconstructor):
        """Check if checkpoint file exists (under `self.models_dir`) and set starting iteration at the checkpoint
        iteration; also load checkpoint weights to `deformator` and `reconstructor`. Otherwise, set starting
        iteration to 1 in order to train from scratch.

        Returns:
            starting_iter (int): starting iteration

        """
        starting_iter = 1
        if osp.isfile(self.checkpoint):
            checkpoint_dict = torch.load(self.checkpoint)
            starting_iter = checkpoint_dict['iter']
            deformator.load_state_dict(checkpoint_dict['deformator'])
            reconstructor.load_state_dict(checkpoint_dict['reconstructor'])
        return starting_iter

    def log_progress(self, iteration, mean_iter_time, elapsed_time, eta):
        """Log progress in terms of batch accuracy, classification and regression losses and ETA.

        Args:
            iteration (int)        : current iteration
            mean_iter_time (float) : mean iteration time
            elapsed_time (float)   : elapsed time until current iteration
            eta (float)            : estimated time of experiment completion

        """
        # Get current training stats (for the previous `self.params.log_freq` steps) and flush them
        stats = self.stat_tracker.get_means()

        # Update training statistics json file
        with open(self.stats_json) as f:
            stats_dict = json.load(f)
        stats_dict.update({iteration: stats})
        with open(self.stats_json, 'w') as out:
            json.dump(stats_dict, out)

        # Flush training statistics tracker
        self.stat_tracker.flush()

        update_progress("  \\__.Training [bs: {}] [iter: {:06d}/{:06d}] ".format(
            self.params.batch_size, iteration, self.params.max_iter), self.params.max_iter, iteration + 1)
        if iteration < self.params.max_iter - 1:
            print()
        print("      \\__Batch accuracy      : {:.03f}".format(stats['accuracy']))
        print("      \\__Classification loss : {:.08f}".format(stats['classification_loss']))
        print("      \\__Regression loss     : {:.08f}".format(stats['regression_loss']))
        print("      \\__Penalty loss        : {:.08f}".format(stats['penalty_loss']))
        print("      \\__Total loss          : {:.08f}".format(stats['total_loss']))
        print("         ===================================================================")
        print("      \\__Mean iter time      : {:.3f} sec".format(mean_iter_time))
        print("      \\__Elapsed time        : {}".format(sec2dhms(elapsed_time)))
        print("      \\__ETA                 : {}".format(sec2dhms(eta)))
        print("         ===================================================================")
        update_stdout(10)

    def train(self, generator, deformator, reconstructor):
        """Training function.

        Args:
            generator     :
            deformator  :
            reconstructor :

        """
        # Save initial `deformator` model as `support_sets_init.pt`
        torch.save(deformator.state_dict(), osp.join(self.models_dir, 'deformator_init.pt'))

        # Set `generator` to evaluation mode, `deformator` and `reconstructor` to training mode, and upload
        # models to GPU if `self.use_cuda` is set (i.e., if args.cuda and torch.cuda.is_available is True).
        if self.use_cuda:
            generator.cuda().eval()
            deformator.cuda().train()
            reconstructor.cuda().train()
        else:
            generator.eval()
            deformator.train()
            reconstructor.train()

        # Set support sets optimizer
        deformator_optim = torch.optim.Adam(deformator.parameters(), lr=self.params.deformator_lr)

        # Set shift predictor optimizer
        reconstructor_optim = torch.optim.Adam(reconstructor.parameters(), lr=self.params.reconstructor_lr)

        # Get starting iteration????????????
        starting_iter = self.get_starting_iteration(deformator, reconstructor)

        # Parallelize `generator` and `reconstructor` into multiple GPUs, if available and `multi_gpu=True`.
        if self.multi_gpu:
            print("#. Parallelize G, R over {} GPUs...".format(torch.cuda.device_count()))
            generator = DataParallelPassthrough(generator)
            reconstructor = DataParallelPassthrough(reconstructor)
            cudnn.benchmark = True

        # Check starting iteration
        if starting_iter == self.params.max_iter:
            print("#. This experiment has already been completed and can be found @ {}".format(self.wip_dir))
            print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
            try:
                shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'),dirs_exist_ok="true")
                print("  \\__Done!")
            except IOError as e:
                print("  \\__Already exists -- {}".format(e))
            sys.exit()
        print("#. Start training from iteration {}".format(starting_iter))

        # Get experiment's start time
        t0 = time.time()

        # Start training
        for iteration in range(starting_iter, self.params.max_iter + 1):

            # Get current iteration's start time
            iter_t0 = time.time()

            # Set gradients to zero
            generator.zero_grad()
            deformator.zero_grad()
            reconstructor.zero_grad()

            # Sample latent codes from standard (truncated) Gaussian -- torch.Size([batch_size, generator.dim_z])
            z = sample_z(batch_size=self.params.batch_size, dim_z=generator.dim_z, truncation=self.params.z_truncation)
            if self.use_cuda:
                z = z.cuda()

            # Generate images for the given latent codes
            img = generator(z)

            # Sample indices of shift vectors (`self.params.batch_size` out of `self.params.num_support_sets`)
            target_indices, shifts, basis_shift = self.make_shifts(deformator.input_dim)

            # Deformation
            shift = deformator(basis_shift)
            # Generate images the shifted latent codes
            img_shifted = generator(z, shift)

            # Predict support sets indices and shift magnitudes
            predicted_direction_indices, predicted_shift_magnitudes = reconstructor(img, img_shifted)

            # Calculate classification (cross entropy) and regression (mean absolute error) losses
            classification_loss = self.cross_entropy(predicted_direction_indices, target_indices)
            penalty = hessian_penalty(generator, z, path_idx=basis_shift, G_z=None, path_matrix=deformator, multiple_layers=False,dir_count=self.params.directions_count).mean()
            regression_loss = torch.mean(torch.abs(predicted_shift_magnitudes - shifts))

            # Calculate total loss and back-propagate
            loss = self.params.lambda_cls * classification_loss + self.params.lambda_reg * regression_loss+self.params.lambda_pen *penalty
            loss.backward()

            # Perform optimization step (parameter update)
            deformator_optim.step()
            reconstructor_optim.step()

            # Update statistics tracker
            self.stat_tracker.update(accuracy=torch.mean((torch.argmax(predicted_direction_indices, dim=1) ==
                                                          target_indices).to(torch.float32)).detach(),
                                     classification_loss=classification_loss.item(),
                                     regression_loss=regression_loss.item(),
                                     penalty_loss=penalty.item(),
                                     total_loss=loss.item())

            # Update tensorboard plots for training statistics
            if self.tensorboard:
                for key, value in self.stat_tracker.get_means().items():
                    self.tb_writer.add_scalar(key, value, iteration)

            # Get time of completion of current iteration
            iter_t = time.time()

            # Compute elapsed time for current iteration and append to `iter_times`
            self.iter_times = np.append(self.iter_times, iter_t - iter_t0)

            # Compute elapsed time so far
            elapsed_time = iter_t - t0

            # Compute rolling mean iteration time
            mean_iter_time = self.iter_times.mean()

            # Compute estimated time of experiment completion??????????????????
            eta = elapsed_time * ((self.params.max_iter - iteration) / (iteration - starting_iter + 1))

            # Log progress in stdout
            if iteration % self.params.log_freq == 0:
                self.log_progress(iteration, mean_iter_time, elapsed_time, eta)

            # Save checkpoint model file and deformator / reconstructor model state dicts after current iteration
            if iteration % self.params.ckp_freq == 0:
                # Build checkpoint dict
                checkpoint_dict = {
                    'iter': iteration,
                    'deformator': deformator.state_dict(),
                    'reconstructor': reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict()
                }
                torch.save(checkpoint_dict, self.checkpoint)
        # === End of training loop ===

        # Get experiment's total elapsed time
        elapsed_time = time.time() - t0

        # Save final support sets model
        deformator_model_filename = osp.join(self.models_dir, 'deformator.pt')
        torch.save(deformator.state_dict(), deformator_model_filename)

        # Save final shift predictor model
        reconstructor_model_filename = osp.join(self.models_dir, 'reconstructor.pt')
        torch.save(reconstructor.module.state_dict() if self.multi_gpu else reconstructor.state_dict(),
                   reconstructor_model_filename)

        for _ in range(10):
            print()
        print("#.Training completed -- Total elapsed time: {}.".format(sec2dhms(elapsed_time)))

        print("#. Copy {} to {}...".format(self.wip_dir, self.complete_dir))
        try:
            shutil.copytree(src=self.wip_dir, dst=self.complete_dir, ignore=shutil.ignore_patterns('checkpoint.pt'))
            print("  \\__Done!")
        except IOError as e:
            print("  \\__Already exists -- {}".format(e))
