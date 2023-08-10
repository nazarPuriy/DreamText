from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
import torchvision
from descriptor import Descriptor

# suppress partial model loading warning
logging.set_verbosity_error()

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass



class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    """
    Used to fix the random seeds in order to obtain always the same generation.
    Args:
        seed: number representing the seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, option=0, opt=None):
        """
        Stable Diffusion model.
        Args:
            device: the device used to make the computus. Can be 'GPU' or 'CPU'
            sd_version: the version of stable diffusion used.
            hf_key: can be used instead of the version to use a concrete build.
            option: option representing the running configuration.
        """
        super().__init__()
        self.option = option

        if option != 0:
            print("\n\n--------- OPTION ", option, " ---------\n")
        self.i = 0
        self.device = device
        self.sd_version = sd_version
        self.general_token = None
        self.loaded = False
        self.opt = opt
        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=precision_t)

        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')
            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)


        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        if self.opt.load_token == 1:
            data = json.load(open("./" + self.opt.workspace + "/descriptor/status.txt"))
            self.placeholder_token_id = data["placeholder"]
            self.token_path = "./" + self.opt.workspace + "/descriptor/tokens/"+data["noun"]+"_"+str(data["iteration"])+".pt"
            if self.opt.verbose: print("Loading trained token: ", self.token_path)
            self.text = Descriptor.get_sentence(data['words_list'], data['noun'], data['quantifier'])
            if self.opt.verbose: print("Sentence is: " + self.text)
        elif self.opt.load_token == 2:
            self.placeholder_token_id = [int(num) for num in self.opt.placeholder_token_id[1:-1].split(",")]
            self.token_path = self.opt.token_path

        if self.opt.load_token == 1:
            self.orig_embedding = self.text_encoder.get_input_embeddings().weight[self.placeholder_token_id]
            with torch.no_grad():
                self.new_embedding = torch.load(self.token_path).half()
            self.new_loaded = False



        print(f'[INFO] loaded stable diffusion!')

    def trained_embedding(self):
        if self.new_loaded:
            return
        self.new_loaded = True
        with torch.no_grad():
            embeddings = self.text_encoder.get_input_embeddings()
            embeddings.weight[self.placeholder_token_id] = self.new_embedding
            self.text_encoder.set_input_embeddings(embeddings)

    def base_embedding(self):
        if not self.new_loaded:
            return
        self.new_loaded = False
        with torch.no_grad():
            embeddings = self.text_encoder.get_input_embeddings()
            embeddings.weight[self.placeholder_token_id] = self.orig_embedding
            self.text_encoder.set_input_embeddings(embeddings)

    def get_text_embeds(self, prompt, negative_prompt):
        """
        This method starts with the text prompts and returns the text embeddings. Used in prompt to image.
        Args:
            prompt: The prompt
            negative_prompt: Negative or blank prompt.
        Returns:
        """

        if self.opt.load_token == 1:
            self.trained_embedding()
            prompt = self.text + ", " +  prompt[0].split(",")[1]

        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        with torch.no_grad():
            text_embeddings_upgraded = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings_upgraded])


    def denoise(self, noised_samples, noise, timesteps):
        """
        This method is not used in the Dream Fusion process. This is used to remove the noise
        and produce the image at time step t at once. THis is not diffusion process.
        This is more to understand the underlying behaviour of the diffusion model.
        Args:
            noised_samples: samples with noise
            noise: the amount of noise that has been added.
            timesteps: the time in which the noise has been added.

        Returns:
        """
        alphas_cumprod = self.scheduler.alphas_cumprod

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(noised_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noised_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        clean_samples = (noised_samples - sqrt_one_minus_alpha_prod * noise)/sqrt_alpha_prod
        return clean_samples

    def train_step(self, text_embeddings, pred_rgb, angles, guidance_scale=100):
        """
        This is the training step of the sd model. It recieves a text
        Args:
            text_embeddings: the embedding of the text imput as prompt
            pred_rgb: the predicted image by the NeRF model
            guidance_scale: the guidance scale, a float.
        Returns:
        """

        # Resize to feed into the VAE
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # Encode the image
        latents = self.encode_imgs(pred_rgb_512)

        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        guidance_scale = self.opt.g

        noise_pred_uncond, noise_pred_upgraded = noise_pred.chunk(2)
        noise_pred = noise_pred_upgraded + guidance_scale * (noise_pred_upgraded - noise_pred_uncond)


        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        loss = SpecifyGradient.apply(latents, grad)

        return loss 

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                #torch.save(t, "produce_latents_t.pt")
                #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        """
        Decode the latents into images
        Args:
            latents: latents of the images
        Returns:
        """
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        """
        Encodes the images into lattent, reducing the shape.
        Args:
            imgs: images batchm, 3, 512,512
        Returns:
        """
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




