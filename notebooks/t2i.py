import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from PIL import Image
import yaml
import torch
import torchvision
import clip
import torch.nn.functional as F

from notebook_utils import TextEncoder, load_model, get_generated_images_by_texts


# load stage 1 model: RQ-VAE
vqvae_path = '/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/ffhq_sota/stage1/model.pt'
model_vqvae, _ = load_model(vqvae_path)


# load stage 2 model: RQ-Transformer
model_path = '/ssd_scratch/cvit/sayandebroy/ffhq256-rqvae-8x8x4/21052024_141001/ffhq_sota/stage2/model.pt'
model_ar, config = load_model(model_path, ema=False)


# move models from cpu to gpu
model_ar = model_ar.cuda().eval()
model_vqvae = model_vqvae.cuda().eval()


# the checkpoint of CLIP will be downloaded at the first time.
model_clip, preprocess_clip = clip.load("ViT-B/32", device='cpu')
model_clip = model_clip.cuda().eval()


# prepare text encoder to tokenize natual languages
text_encoder = TextEncoder(tokenizer_name=config.dataset.txt_tok_name, 
                           context_length=config.dataset.context_length)


text_prompts = 'a painting of "the persistence of memory"' # your own text
num_samples = 64
temperature= 1.0
top_k=1024
top_p=0.95


pixels = get_generated_images_by_texts(model_ar,
                                       model_vqvae,
                                       text_encoder,
                                       model_clip,
                                       preprocess_clip,
                                       text_prompts,
                                       num_samples,
                                       temperature,
                                       top_k,
                                       top_p,
                                      )


num_visualize_samples = 16
images = [pixel.cpu().numpy() * 0.5 + 0.5 for pixel in pixels]
images = torch.from_numpy(np.array(images[:num_visualize_samples]))
images = torch.clamp(images, 0, 1)
grid = torchvision.utils.make_grid(images, nrow=4)

img = Image.fromarray(np.uint8(grid.numpy().transpose([1,2,0])*255))
#display(img)


img.save(f'{text_prompts}_temp_{temperature}_top_k_{top_k}_top_p_{top_p}.jpg')