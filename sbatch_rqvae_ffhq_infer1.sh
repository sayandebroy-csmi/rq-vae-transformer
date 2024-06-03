#!/bin/bash
#SBATCH -A sayandebroy.csmi
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist gnode049
#SBATCH --mem-per-cpu=2G
#SBATCH --time=8-00:00:00
#SBATCH --output=output_ffhq_infer.txt
#SBATCH --mail-type=ALL



conda init bash
source activate rqvae

which python

#python main_sampling_fid.py -v=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/ffhq_sota/stage1/model.pt -a=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/ffhq_sota/stage2/model.pt --save-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch25_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_25_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch30_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_30_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch35_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_35_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch40_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_40_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch45_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_45_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch50_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_50_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch55_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_55_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch60_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_60_epoch --eval

python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch65_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_65_epoch --eval

#python inference_rqvae.py -m=configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml -r=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/31052024_123129 -l=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/epoch70_model.pt --output-dir=/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/outputs_70_epoch --eval
