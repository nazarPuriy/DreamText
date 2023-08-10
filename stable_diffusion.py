import os
import argparse
import torch
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer,  CLIPProcessor, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from accelerate import Accelerator


class StableDiffusion:
    def __init__(self, workspace="example", sentence="", device="cuda", model_key = "stabilityai/stable-diffusion-2-1-base"):

        # Create the workspace in where the images will be stored.
        if not os.path.exists(workspace):
            os.mkdir(workspace)
            os.mkdir(workspace + "/images/")

        # Load all the models
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.alphas_cum = self.scheduler.alphas_cumprod.to(device)
        self.alphas = self.scheduler.alphas.to(device)
        self.accelerator = Accelerator()

        # Set them as non-trainable
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.device = device
        self.sentence = sentence
        self.workspace = workspace

    def decode_latents(self, latents):
        """
        Decode the latents into images
        Args:
            latents: latents of the images
        Returns: images
        """
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        return (imgs / 2 + 0.5).clamp(0, 1)

    def sample(self, number=5):
        """
        Generates samples (images) by utilizing trained tokens with the obtained sentence and stores them.

        Args:
            number (int, optional): The number of samples to be generated. Default value is 5.

        Returns:
            None. Stores image in file system under the directory ./workspace/descriptor/samples
        """

        # First load the text embedding calculated from the model that is trained.
        text_input = self.tokenizer("", padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        with torch.no_grad():
            no_info = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_input = self.tokenizer(self.sentence, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        with torch.no_grad():
            embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_embed = torch.cat([no_info, embed])

        for i in range(number):
            # Sample random noise
            xt = torch.randn([1, 4, 64, 64]).to(self.device)

            with torch.no_grad():
                guidance = 10  # Indicates how much impact does the text have on the generation
                step = 10  # Indicate how much diffusion backward steps are done altogether
                alphas_cum = self.alphas_cum
                for t in range(999, 0, -step):
                    xt = torch.cat([xt] * 2)

                    # First predict the noise using the text embedding
                    with torch.no_grad():
                        noise_pred = self.unet(xt, t, encoder_hidden_states=text_embed).sample
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_text + guidance * (noise_pred_text - noise_pred_uncond)

                    # Generate the previous, less nosied, version of the image
                    if t - step > 0:
                        xt = (alphas_cum[t - step] ** 0.5) * (xt[0] - ((1 - alphas_cum[t]) ** 0.5) * noise_pred) / (
                                alphas_cum[t] ** 0.5) + ((1 - alphas_cum[t - step]) ** 0.5) * noise_pred
                    else:
                        xt = (xt[0] - ((1 - alphas_cum[t]) ** 0.5) * noise_pred) / (alphas_cum[t] ** 0.5)

                    # If this is the last step store the image
                    if t - step < 1:
                        img = self.decode_latents(xt)
                        save_image(img, "./" + self.workspace + "/images/image_" + str(
                            i) + ".png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default='example', type=str, required=True, help="Where the files will be stored")
    parser.add_argument('--text', type=str, default="", required=True, help="Text used to generate the images")
    parser.add_argument('--number', default=5, type=int)
    opt = parser.parse_args()


    sd = StableDiffusion(workspace=opt.workspace, sentence=opt.text)
    sd.sample()


