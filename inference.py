import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_utils import distributed as dist
import dnnlib
from training import dataset
from torch_utils.misc import StackedRandomGenerator
from torch_utils.ambient_diffusion import get_well_mask
import json
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import argparse
import colorcet as cc
import pdb

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def cdist_masked(x1, x2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(x1)
        mask2 = torch.ones_like(x2)
    x1 = x1[0].unsqueeze(0)
    diffs = x1.unsqueeze(1) - x2.unsqueeze(0)
    combined_mask = mask1.unsqueeze(1) * mask2.unsqueeze(0)
    error = 0.5 * torch.linalg.norm(combined_mask * diffs)**2
    return error

def ambient_sampler(
    net, latents, randn_like=torch.randn_like,
    num_steps=30, sigma_min=0.05, sigma_max=80, rho=7,
    S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=1,
    sampler_seed=42, 
    same_for_all_batch=False,
    num_masks=1,
    guidance_scale=0.0,
    clipping=True,
    static=True,  # whether to use soft clipping or static clipping
    resample_guidance_masks=False,
    rtm_loc = "", guidance_weight = 0.0
):
    class_labels = None
    # sigma_min = max(sigma_min, net.sigma_min)
    # sigma_max = min(sigma_max, net.sigma_max)
    print("net.sigma_min")
    print(sigma_min)
    print("net.sigma_max")
    print(sigma_max)

    clean_image = None

    def sample_masks():
        masks = []
        for i in range(num_masks):
            #corruption_mask = get_well_mask(latents.shape, 256, same_for_all_batch=False, device=latents.device, seed=sampler_seed)
            corruption_mask = get_well_mask(latents.shape, 4, same_for_all_batch=False, device=latents.device, seed=None)
            masks.append(corruption_mask)

            # corruption_mask = torch.from_numpy(torch.load("corruption_masks/hat_corruption_mask_0.pt")).unsqueeze(0).unsqueeze(0).to(('cuda'))
            # masks.append(corruption_mask)

            # plt.figure()
            # plt.imshow(corruption_mask.detach().cpu().numpy()[0,0,:,:])
            # plt.colorbar()
            # plt.savefig("corruption_mask_{}".format(i))
            # plt.close()
        masks = torch.stack(masks)
        return masks

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    print("step_indices")
    print(step_indices)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    print("t_steps")
    print(t_steps)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    print("x_next")
    print(x_next)


    uncond = torch.zeros((1, 256, 256)) 
    uncond = uncond.repeat(1,1,1,1).to(('cuda'))
    #Deliberately throw an error
    # raise Exception("An error occurred after the first line.")

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # masks = sample_masks()
        # print("t_next")
        # print(t_next)
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        # print("t_hat")
        # print(t_hat)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        # x_hat = x_cur
        x_hat = x_hat.detach()
        x_hat.requires_grad = True

        denoised = []
        for mask_index in range(num_masks):
            print(mask_index)
            masks = sample_masks()
            print(masks.shape)
            corruption_mask = masks[mask_index]

            masked_image = corruption_mask * x_hat
            # masked_image = masked_image/torch.max(torch.abs(masked_image))

            # d_tensor = torch.from_numpy(torch.load(rtm_loc))
            # d_tensor = d_tensor.unsqueeze(0)
            d_tensor = np.load(rtm_loc) 
            d_tensor = torch.from_numpy(d_tensor[np.newaxis,...]) 
            #d_tensor_repeated = d_tensor.repeat(1,1,1,1).to(('cuda'))
            d_tensor_repeated = d_tensor.repeat(1,1,1,1).to(('cuda'))
            # d_tensor_repeated = d_tensor_repeated + 0.01 * randn_like(d_tensor_repeated)
            # print(d_tensor_repeated.shape)
            # print(masked_image.shape)
            # print(corruption_mask.shape)

            net_input_cond = torch.cat([masked_image, corruption_mask, d_tensor_repeated], dim=1)
            net_input_uncond = torch.cat([masked_image, corruption_mask, uncond], dim=1)

            net_output_cond = net(net_input_cond, t_hat, class_labels).to(torch.float64)[:, :1]
            net_output_uncond= net(net_input_uncond, t_hat, class_labels).to(torch.float64)[:, :1]

            net_output = (1 + guidance_weight) * net_output_cond - guidance_weight * net_output_uncond

            # print(net_output.shape)

            #if clipping:
            #    net_output = tensor_clipping(net_output, static=static)

            if clean_image is not None:
                net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image

            # Euler step.
            denoised.append(net_output)


        stack_denoised = torch.stack(denoised)
        flattened = stack_denoised.view(stack_denoised.shape[0], -1)
        l2_norm = cdist_masked(flattened, flattened, None, None)
        l2_norm = l2_norm.mean()
        rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]

        # print("Stack Denoised Shape")
        # print(stack_denoised.shape)
        # clean_pred = stack_denoised[0]
        clean_pred = torch.mean(stack_denoised, dim=0, keepdim=True).squeeze(0)
        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        grad_1 = single_mask_grad - guidance_scale * rec_grad
        x_next += grad_1

        if i < num_steps - 1:
            masks = sample_masks()
            x_next = x_next.detach()
            x_next.requires_grad = True

            denoised = []
            for mask_index in range(num_masks):
                corruption_mask = masks[mask_index]
                masked_image = corruption_mask * x_next
                # masked_image = masked_image/torch.max(torch.abs(masked_image))

                # plt.figure()
                # plt.imshow(masked_image.detach().cpu().numpy()[0,0,:,:])
                # plt.savefig("masked_images_iterated_32_network/masked_image{}".format(i))
                # plt.close()


                net_input_cond = torch.cat([masked_image, corruption_mask, d_tensor_repeated], dim=1)
                net_input_uncond = torch.cat([masked_image, corruption_mask, uncond], dim=1)

                net_output_cond = net(net_input_cond, t_hat, class_labels).to(torch.float64)[:, :1]
                net_output_uncond= net(net_input_uncond, t_hat, class_labels).to(torch.float64)[:, :1]

                net_output = (1 + guidance_weight) * net_output_cond - guidance_weight * net_output_uncond

                #if clipping:
                #    net_output = tensor_clipping(net_output, static=static)
                
                if clean_image is not None:
                    net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image
                denoised.append(net_output)
            
            stack_denoised = torch.stack(denoised)
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None)
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_next)[0]

            clean_pred = torch.mean(stack_denoised, dim=0, keepdim=True).squeeze(0)
            single_mask_grad = (t_next - t_hat) * (x_next - clean_pred) / t_next
            grad_2 = single_mask_grad - guidance_scale * rec_grad
            x_next = x_hat + 0.5 * (grad_1 + grad_2)

            # plt.figure()
            # plt.imshow(x_next.detach().cpu().numpy()[0,0,:,:])
            # #plt.savefig("x_next_iterated_32_network/x_next_{}".format(i))
            # plt.savefig("5_well_good_rtm_1040_dataset_sample/5well_good_rtm_test_8242_2/8242_test_rtm2/x_next/x_next_{}".format(i))
            # plt.close()
        else:
            if clean_image is not None:
                x_next = masks[0] * x_next + (1 - masks[0]) * clean_image
            else:
                clean_image = x_next
                x_next = x_hat + grad_1
            # plt.figure()
            # plt.imshow(x_next.detach().cpu().numpy()[0,0,:,:])
            # #plt.savefig("x_next_iterated_32_network/x_next_{}".format(i))
            # plt.savefig("5_well_good_rtm_1040_dataset_sample/5well_good_rtm_test_8242_2/8242_test_rtm2/x_next/x_next_{}".format(i))
            # plt.close()
    return x_next

def main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc, gt_loc, device=torch.device('cuda'),  **sampler_kwargs):

    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # we want to make sure that each gpu does not get more than batch size.
    # Hence, the following measures how many batches are going to be per GPU.
    seeds = seeds[:num_generate]
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    print(num_batches)
    dist.print0(f"The algorithm will run for {num_batches} batches --  {len(seeds)} images of batch size {max_batch_size}")
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    # the following has for each batch size allocated to this GPU, the indexes of the corresponding images.
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    batches_per_process = len(rank_batches)
    dist.print0(f"This process will get {len(rank_batches)} batches.")

    # load training options
    with dnnlib.util.open_url(training_options_loc, verbose=(dist.get_rank() == 0)) as f:
        training_options = json.load(f)

    if training_options['dataset_kwargs']['use_labels']:
        assert num_classes > 0, "If the network is class conditional, the number of classes must be positive."
        label_dim = num_classes
    else:
        label_dim = 0

    interface_kwargs = dict(img_resolution=training_options['dataset_kwargs']['resolution'], label_dim=label_dim, img_channels=3)
    network_kwargs = training_options['network_kwargs']
    model_to_be_initialized = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module

    # find all *.pkl files in the folder network_loc and sort them
    files = dnnlib.util.list_dir(network_loc)
    # Filter the list to include only "*.pkl" files
    pkl_files = [f for f in files if f.endswith('.pkl')]

    # Sort the list of "*.pkl" files
    sorted_pkl_files = sorted(pkl_files)
    sorted_pkl_files = [sorted_pkl_files[-1]] # use only the most recent network

    checkpoint_numbers = []
    for curr_file in sorted_pkl_files:
        checkpoint_numbers.append(int(curr_file.split('-')[-1].split('.')[0]))
    checkpoint_numbers = np.array(checkpoint_numbers)
    
    for checkpoint_number, checkpoint in zip(checkpoint_numbers, sorted_pkl_files):
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        network_pkl = os.path.join(network_loc, f'network-snapshot-{checkpoint_number:06d}.pkl')
        # Load network.
        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            loaded_obj = pickle.load(f)['ema']
        
        if type(loaded_obj) == OrderedDict:
            COMPILE = False
            if COMPILE:
                net = torch.compile(model_to_be_initialized)
                net.load_state_dict(loaded_obj)
            else:
                modified_dict = OrderedDict({key.replace('_orig_mod.', ''):val for key, val in loaded_obj.items()})
                net = model_to_be_initialized
                net.load_state_dict(modified_dict)
        else:
            # ensures backward compatibility for times where net is a model pkl file
            net = loaded_obj
        net = net.to(device)
        dist.print0(f'Network loaded!')

        #pdb.set_trace()

        image_dir = os.path.join(outdir, str(checkpoint_number) + "/" + cond_loc[-12:-4])
        os.makedirs(image_dir, exist_ok=True)

        cond = np.load(cond_loc)
        cond = torch.from_numpy(cond) 
        cond = cond.repeat(1,1,1,1).to((device))

        cond[0,0,0:16,:] = 0
       
        a = np.quantile(np.absolute(cond.cpu()),0.98)
        plt.figure(); plt.title("Condition")
        plt.imshow(cond[0,0,:,:].cpu(), vmin=-a,vmax=a, cmap = "gray")
        plt.axis("off")
        cb = plt.colorbar(fraction=0.0235, pad=0.04); 
        plt.savefig(os.path.join(image_dir, "actual_condition.png"),bbox_inches = "tight",dpi=300)

        gt = np.load(gt_loc) 
        vmin_gt = None#1.5
        vmax_gt = None#3.7#4.5
        cmap_gt = cc.cm['rainbow4']

        plt.figure();  plt.title("Ground truth")
        plt.imshow(gt, vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
        plt.axis("off")
        cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
        plt.savefig(os.path.join(image_dir, "original_velocity.png"),bbox_inches = "tight",dpi=300)

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
        batch_count = 1
        images_np_stack = np.zeros((len(seeds),1,*gt.shape))
        for batch_seeds in tqdm.tqdm(rank_batches, disable=dist.get_rank() != 0):
            dist.print0(f"Waiting for the green light to start generation for {batch_count}/{batches_per_process}")
            # don't move to the next batch until all nodes have finished their current batch
            torch.distributed.barrier()
            dist.print0("Others finished. Good to go!")
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, 1, gt.shape[0], gt.shape[1]], device=device)
           
            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            # images = ambient_sampler(net, latents,num_steps=num_steps, randn_like=rnd.randn_like,
            #     cond=cond, image_dir=image_dir, **sampler_kwargs)
            images = ambient_sampler(net, latents, randn_like=rnd.randn_like, sampler_seed=batch_seeds, 
                guidance_scale=0.0, 
                rtm_loc = cond_loc, guidance_weight = 0.0, **sampler_kwargs,)
            
            # Save Images
            images_np = images.cpu().detach().numpy()
            for seed, one_image in zip(batch_seeds, images_np):
                dist.print0(f"Saving loc: {image_dir}")
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, "steps_"+str(num_steps)+"_"+f'{seed:04d}.png')

                plt.figure(); plt.title("Posterior Sample")
                plt.imshow(one_image[0, :, :],   vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
                plt.axis("off")
                cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
                plt.savefig(image_path, bbox_inches = "tight",dpi=300)
                plt.close()
                
                os.makedirs(os.path.join(image_dir, f'saved/'), exist_ok=True)
                np.save(os.path.join(image_dir, f'saved/{seed:06d}')+ ".npy", one_image[0, :, :])
            images_np_stack[batch_count-1,0,:,:] = one_image
            batch_count += 1

           # plot posterior statistics
        post_mean = np.mean(images_np_stack,axis=0)[0,:,:]
        ssim_t = ssim(gt,post_mean, data_range=np.max(gt) - np.min(gt))

        plt.figure(); plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
        plt.imshow(post_mean,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt)
        plt.axis("off"); 
        cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_mean.png"),bbox_inches = "tight",dpi=300); plt.close()

        plt.figure(); plt.title("Stdev")
        plt.imshow(np.std(images_np_stack,axis=0)[0,:,:],  vmin=0, vmax=None,   cmap = "magma")
        plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"std.png"),bbox_inches = "tight",dpi=300); plt.close()
            
        rmse_t = np.sqrt(mean_squared_error(gt, post_mean))
        plt.figure(); plt.title("Error RMSE:"+str(round(rmse_t,4)))
        plt.imshow(np.abs(post_mean-gt), vmin=0, vmax=None, cmap = "magma")
        plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
        plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_error.png"),bbox_inches = "tight",dpi=300); plt.close()

        dist.print0(f"Node finished generation for {checkpoint_number}")
        dist.print0("waiting for others to finish..")

        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        dist.print0("Everyone finished.. Starting calculation..")
    
if __name__ == "__main__":
   
    seeds = [i for i in range(0, 100)]
    max_batch_size = 1
    num_generate = 2
    num_steps = 10

    device = torch.device('cuda')
    #device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_loc', type=str, default="")
    parser.add_argument('--network_loc', type=str, default="")
    parser.add_argument('--gt_loc', type=str, default="")

    args = parser.parse_args()
    cond_loc = args.cond_loc
    vel_loc = args.gt_loc
    network_loc = args.network_loc

    training_options_loc = network_loc+"/training_options.json"
    outdir = "sampling/"

    main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc, vel_loc, device)
    