# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# 
# Modified July 18, 2023
# ---------------------------------------------------------------

import argparse
import torch
import torch.nn as nn
import numpy as np
import os

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from vae.model import AutoEncoder
from thirdparty.adamax import Adamax
import vae.utils as utils
import vae.datasets as datasets


import matplotlib.pyplot as plt

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Get data loaders.
    train_queue, valid_queue = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, writer, arch_instance)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset) # total number of output pixels
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)


        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, 
                                         global_step, warmup_iters, writer, logging,
                                        )
        
        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)

        model.eval()
        # generate samples less frequently
        eval_freq = 1 if args.epochs <= 50 else 20
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):

            valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=10, args=args, logging=logging)
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)

        save_freq = int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()}, checkpoint_file)


    # Final validation
    valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=10, args=args, logging=logging, save_images=True)
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)

    writer.close()

def parse_x_full(x_full, args):
    # x_full is (sparse_reconstruction, sparse_sinogram, sparse_sinogram_raw, object_id,
    # angles, x_size, y_size, num_proj_pix, ground_truth)
    x = x_full[0].cuda()
    sparse_sinogram_raw = x_full[2].cuda()
    sparse_sinogram = x_full[1].cuda()
    ground_truth = x_full[8].cuda()
    theta = x_full[4].cuda()
    x_size = x_full[5].cuda()
    # change bit length
    x = utils.pre_process(x, args.num_x_bits)
    return(x, sparse_sinogram_raw, sparse_sinogram, ground_truth, theta, x_size)

def process_decoder_output(x, args, model, theta,
                           sparse_sinogram_raw, x_size):

    logits, log_q, log_p, kl_all, kl_diag = model(x)
    temperature = torch.tensor([args.temp_bernoulli])
    temperature = temperature.half().cuda()

    theta_degrees = theta*180/np.pi
    sino_raw_dist, phantom = model.decoder_output(logits, temperature, theta_degrees=theta_degrees.half(), 
                                                  poisson_noise_multiplier=args.pnm, pad=False, normalizer=x_size) 
    recon_loss = utils.reconstruction_loss(sino_raw_dist, sparse_sinogram_raw, args.dataset, crop=model.crop_output)
    return(logits, log_q, log_p, kl_all, kl_diag, 
           sino_raw_dist, phantom, recon_loss)

def train(train_queue, model, cnn_optimizer, grad_scalar, 
          global_step, warmup_iters, writer, logging,
          ):

    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, x_full in enumerate(train_queue):
        x, sparse_sinogram_raw, sparse_sinogram, ground_truth, theta, x_size = parse_x_full(x_full, args)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag, \
            sino_raw_dist, phantom, recon_loss = \
                process_decoder_output(x, args, model, theta, sparse_sinogram_raw, x_size)
            
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            
            loss = torch.mean(nelbo_batch)
            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 5 == 0:
            if (global_step + 1) % 20 == 0:  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))

                x_img = x[:n*n]
                x_tiled = utils.tile_image(x_img, n)

                writer.add_image('input image', x_tiled, global_step)
                plt.figure()
                plt.imshow(x_tiled[0].detach().cpu().numpy())
                plt.savefig(args.save + '/input_image_' + str(global_step)+'.png')

                ground_truth = ground_truth[:n*n,None]
                ground_truth_tiled = utils.tile_image(ground_truth, n)
                writer.add_image('ground truth', ground_truth_tiled, global_step)
                plt.figure()
                plt.imshow(ground_truth_tiled[0].detach().cpu().numpy())
                plt.savefig(args.save + '/ground_truth_' + str(global_step)+'.png')

                output_sinogram_raw = sino_raw_dist.mean if isinstance(sino_raw_dist, torch.distributions.bernoulli.Bernoulli) else sino_raw_dist.sample()
                output_sinogram_raw = output_sinogram_raw[:n*n]
                output_sinogram_raw = output_sinogram_raw[:,None,:,:]
                output_sinogram_raw = utils.tile_image(output_sinogram_raw, n)  

                sparse_sinogram_raw = sparse_sinogram_raw[:n*n]
                sparse_sinogram_raw = sparse_sinogram_raw[:,None,:,:]
                sparse_sinogram_tiled = utils.tile_image(sparse_sinogram_raw, n)  
                in_out_tiled = torch.cat((sparse_sinogram_tiled, output_sinogram_raw), dim=2)
                # breakpoint()   
                writer.add_image('sinogram reconstruction', in_out_tiled, global_step)
                plt.figure()
                plt.imshow(in_out_tiled[0].detach().cpu().numpy())
                plt.savefig(args.save + '/sinogram_reconstruction_' + str(global_step)+'.png')

                phantom_sample = phantom
                phantom_sample = phantom_sample[:n*n]
                phantom_sample = torch.transpose(phantom_sample,2,3)
                phantom_sample = torch.transpose(phantom_sample,1,2)
                phantom_tiled = utils.tile_image(phantom_sample, n)

                writer.add_image('phantom_reconstruction', phantom_tiled, global_step)
                plt.figure()
                plt.imshow(phantom_tiled[0].detach().cpu().numpy())
                plt.savefig(args.save + '/phantom_reconstruction_' + str(global_step)+'.png')
                plt.close('all')

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(sino_raw_dist, sparse_sinogram_raw, args.dataset, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, num_samples, args, logging, save_images=False):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, x_full in enumerate(valid_queue):
        x, sparse_sinogram_raw, sparse_sinogram, ground_truth, theta, x_size = parse_x_full(x_full, args)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, kl_diag, \
                sino_raw_dist, phantom, recon_loss = \
                    process_decoder_output(x, args, model, theta, sparse_sinogram_raw, x_size)

                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(sino_raw_dist, sparse_sinogram_raw, log_q, log_p, args.dataset, crop=model.crop_output))

            if save_images:
                # save ground truth
                np.save(args.save + '/ground_truth_' + str(step) + '_global_rank_' + str(args.global_rank) + '.npy', ground_truth.cpu().numpy())
                # save phantom
                np.save(args.save + '/final_phantom_' + str(step) + '_global_rank_' + str(args.global_rank) + '.npy', phantom.cpu().numpy())

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg



def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    if args.use_nersc:
        os.environ['MASTER_PORT'] = '29500'
    else:
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='foam',
                        help='dataset type to use, dataset should be in format dataset_type')

    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    parser.add_argument('--temp', dest='temp_bernoulli', type=float, default=2.2,
                        help='temperature of relaxed bernoulli')
    # physics parameters
    parser.add_argument('--pnm', dest='pnm', type=float, default=1e3,
                        help='poisson noise multiplier, higher value means higher SNR')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--use_nersc', action='store_true', default=False,
                        help='This flag is for running on NERSC.')
    
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)


