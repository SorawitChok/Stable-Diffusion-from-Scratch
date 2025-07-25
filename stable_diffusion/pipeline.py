import torch
import numpy as np
from tqdm import tqdm

from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(
    prompt: str,
    neg_prompt: str,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None
):
    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert prompt into token using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch size, Seq len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch size, Seq len) -> (Batch size, Seq len, Dim) = (1, 77, 768)
            cond_context = clip(cond_tokens)
            
            uncond_tokens = tokenizer.batch_encode_plus([neg_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch size, Seq len) -> (Batch size, Seq len, Dim) = (1, 77, 768)
            uncond_context = clip(uncond_tokens)

            # (Batch size, Seq len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert prompt into token using tokenizer
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch size, Seq len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch size, Seq len) -> (Batch size, Seq len, Dim) = (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1, 1))
            # (Height, Width, Channel) -> (Batch size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch size, Height, Width, Channel) -> (Batch size, Channel, Width, Height)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise =torch.randn(latent_shape, generator=generator, device=device)
            # run image through the encoder of VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # If doing text-to-image generate random noise from normal distribution N(0, I)
            latents = torch.randn(latent_shape, generator=generator, device=device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch size, 4, LATENT_HEIGHT, LATENT_WIDTH)
            model_input = latents

            if do_cfg:
                # (Batch size, 4, LATENT_HEIGHT, LATENT_WIDTH) -> # (2*Batch size, 4, LATENT_HEIGHT, LATENT_WIDTH)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by UNET
            model_output = diffusion(model_input, context, time_embedding)


            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        image = decoder(latents)
        to_idle(decoder)

        image = rescale(image, (-1, 1), (0, 255), clamp=True)
        # (Btach size, Channels, Height, Width) -> (Batch size, Height, Width, Channels)
        image = image.permute(0, 2, 3, 1)
        image = image.to("cpu", torch.uint8).numpy()
        return image[0]
    
def rescale(inputs: torch.Tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = inputs - old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        
