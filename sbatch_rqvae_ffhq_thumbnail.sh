#!/bin/bash
#SBATCH -A sayandebroy.csmi
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist gnode056
#SBATCH --mem-per-cpu=2G
#SBATCH --time=8-00:00:00
#SBATCH --output=output_rqvae_ffhq_thumbnails_resume_from_epoch70.txt
#SBATCH --mail-type=ALL



conda init bash
source activate rqvae

which python

#python main_stage1.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy

python main_stage1.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch70_model.pt --resume
