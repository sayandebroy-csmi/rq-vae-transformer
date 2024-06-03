import os
import argparse
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from rqvae.utils.utils import set_seed, compute_model_size
from rqvae.models import create_model
from rqvae.img_datasets import create_dataset
from rqvae.utils.setup import setup
import rqvae.utils.dist as dist_utils
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model-config', type=str, default='./configs/c10-igpt.yaml')
parser.add_argument('-r', '--result-path', type=str, default='./results.tmp')
parser.add_argument('-l', '--load-path', type=str, default='')
parser.add_argument('-p', '--postfix', type=str, default='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
parser.add_argument('--eval', action='store_true', help='Run evaluation')
parser.add_argument('--resume', action='store_true')

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--timeout', type=int, default=86400, help='time limit (s) to wait for other nodes in DDP')

args, extra_args = parser.parse_known_args()

set_seed(args.seed)


def save_output(output, output_dir, index):
    # Remove the batch dimension
    output = output.squeeze(0)

    # Convert the output tensor to a numpy array
    output_np = output.cpu().numpy()
    
    #print(f"Shape of output tensor after squeezing: {output_np.shape}")
    
    # Check if the output tensor has 3 dimensions (C, H, W)
    if len(output_np.shape) == 3:
        # Convert to (H, W, C) format
        output_np = np.transpose(output_np, (1, 2, 0))  
    
    # If the output tensor has 2 dimensions (H, W), it is already in the correct format
    elif len(output_np.shape) != 2:
        raise ValueError(f"Unexpected output tensor shape: {output_np.shape}")
    
    # Normalize the output tensor to the range [0, 1]
    output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())

    # Convert the numpy array to an image
    output_img = Image.fromarray((output_np * 255).astype(np.uint8))
    
    # Save the image
    output_path = os.path.join(output_dir, f'output_{index}.png')
    output_img.save(output_path)



if __name__ == '__main__':
    # Ensure 'eval' attribute exists before calling setup
    if not hasattr(args, 'eval'):
        setattr(args, 'eval', False)

    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)
    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    if model_ema:
        model_ema = model_ema.to(device)

    # Load the model weights
    if args.load_path:
        ckpt = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        if model_ema:
            model_ema.load_state_dict(ckpt['state_dict_ema'])
        if distenv.master:
            logger.info(f'{args.load_path} model is loaded')

    # Prepare for distributed data parallel
    model = dist_utils.dataparallel_and_sync(distenv, model)
    if model_ema:
        model_ema = dist_utils.dataparallel_and_sync(distenv, model_ema)

    # Inference
    model.eval()
    if model_ema:
        model_ema.eval()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Running inference on the validation dataset
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset_val, desc="Inference Progress")):
            inputs, labels = data
            inputs = inputs.to(device)

            # Add a batch dimension to the input tensor
            inputs = inputs.unsqueeze(0)

            # Forward pass
            outputs = model(inputs)
            if model_ema:
                outputs_ema = model_ema(inputs)

            # Save the output
            save_output(outputs[0], args.output_dir, i)
            if model_ema:
                save_output(outputs_ema[0], args.output_dir, f'{i}_ema')

    print("Inference Completed")
