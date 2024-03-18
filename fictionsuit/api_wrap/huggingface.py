import transformers
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, EulerDiscreteScheduler
from PIL import Image
import pickle
import base64
import uuid
from ..core.fictionscript import Scope
from ..core.fictionscript.ui_parameter import UiParameter, UiButton

from ..commands.failure import CommandFailure

torch.backends.cuda.matmul.allow_tf32 = True

class DiffusionResult:
    def __init__(self, imageAsTensor):
        # TODO: add settings to result
        self.imageAsTensor = imageAsTensor
        self.cache_id = uuid.uuid4()
    
    def sm_schematize(self, cache):
        cache[str(self.cache_id)] = self
        return {"schema": "image_reference",
                "source": f"{self.cache_id}.png",
                "alt": "alt text goes here",
                "width": 512,
                "height": 512}
    
    def pil(self):
        return Image.fromarray(self.imageAsTensor)

class DiffusionModel:
    def __init__(self, model: str):
        self.model_source = model
        self.vae = AutoencoderKL.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer", torch_dtype=torch.float16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=torch.float16
        )
        if True: # SD XL
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                model, subfolder="tokenizer_2", torch_dtype=torch.float16
            )
            self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model, subfolder="text_encoder_2", torch_dtype=torch.float16
            )
            self.text_encoder_2.to(device="cuda")
        self.unet = UNet2DConditionModel.from_pretrained(
            model, subfolder="unet", torch_dtype=torch.float16
        )

        # Not working with conda env "ml2"
        #self.unet = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", torch_dtype=torch.float16
        )
        self.vae.to(device="cuda")
        self.text_encoder.to(device="cuda")
        self.unet.to(device="cuda")
        self.settings = Scope("diffusion settings")
        self.settings.vars["prompt"] = UiParameter("prompt", "")
        
        self.settings.vars["antiprompt"] = UiParameter("prompt", "")
        self.settings.vars["height"] = UiParameter("int x64", 1024)
        self.settings.vars["width"] = UiParameter("int x64", 1024)
        self.settings.vars["steps"] = UiParameter("int", 125)
        self.settings.vars["scale"] = UiParameter("number", 10)
        async def generate():
            return await self.sm_default("")
        self.settings.vars["go_button"] = UiButton("Generate", generate)

    def sm_schematize(self, cache):
        return self.settings.sm_schematize(cache)

    async def sm_inspect(self):
        return f"Diffusion Model `{self.model_source}`"
    
    async def sm_settings(self, args):
        return self.settings

    async def sm_default(self, args):
        if not isinstance(self.settings.vars["height"].value, int):
            self.settings.vars["height"].value = int(self.settings.vars["height"].value)
            if self.settings.vars["height"].value < 64:
                self.settings.vars["height"].value *= 64
        if not isinstance(self.settings.vars["width"].value, int):
            self.settings.vars["width"].value = int(self.settings.vars["width"].value)
            if self.settings.vars["width"].value < 64:
                self.settings.vars["width"].value *= 64
        if not isinstance(self.settings.vars["steps"].value, int):
            self.settings.vars["steps"].value = int(self.settings.vars["steps"].value)
        if not isinstance(self.settings.vars["scale"], int):
            self.settings.vars["scale"].value = int(self.settings.vars["scale"].value)


        # generator = torch.manual_seed(0)
        batch_size = 1
        positive_tokens = self.tokenizer(
            [self.settings.vars["prompt"].value],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_tokens = self.tokenizer(
            [self.settings.vars["antiprompt"].value],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        positive_tokens_2 = self.tokenizer_2(
            [self.settings.vars["prompt"].value],
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_tokens_2 = self.tokenizer_2(
            [self.settings.vars["antiprompt"].value],
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )



        with torch.no_grad():
            positive_embeddings = self.text_encoder(
                positive_tokens.input_ids.to(device="cuda"),
                output_hidden_states = True
            )
            negative_embeddings = self.text_encoder(
                negative_tokens.input_ids.to(device="cuda"),
                output_hidden_states = True
            )
            positive_embeddings_2 = self.text_encoder_2(
                positive_tokens_2.input_ids.to(device="cuda"),
                output_hidden_states = True
            )
            negative_embeddings_2 = self.text_encoder_2(
                negative_tokens_2.input_ids.to(device="cuda"),
                output_hidden_states = True
            )

            pooled_positives = positive_embeddings_2[0]
            pooled_negatives = negative_embeddings_2[0]

            prompt_embeds = positive_embeddings.hidden_states[-2]
            antiprompt_embeds = negative_embeddings.hidden_states[-2]
            prompt_embeds_2 = positive_embeddings_2.hidden_states[-2]
            antiprompt_embeds_2 = negative_embeddings_2.hidden_states[-2]

            bs_embed, seq_len, _ = prompt_embeds.shape

            prompt_embeds = prompt_embeds.repeat(1,1,1)
            antiprompt_embeds = antiprompt_embeds.repeat(1,1,1)
            prompt_embeds_2 = prompt_embeds_2.repeat(1,1,1)
            antiprompt_embeds_2 = antiprompt_embeds_2.repeat(1,1,1)

            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            antiprompt_embeds = antiprompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_2 = prompt_embeds_2.view(bs_embed, seq_len, -1)
            antiprompt_embeds_2 = antiprompt_embeds_2.view(bs_embed, seq_len, -1)

            pooled_positives = pooled_positives.repeat(1,1,1).view(bs_embed, -1)
            pooled_negatives = pooled_negatives.repeat(1,1,1).view(bs_embed, -1)

            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
            antiprompt_embeds = torch.cat([antiprompt_embeds, antiprompt_embeds_2], dim=-1)

            embeddings = torch.cat([antiprompt_embeds, prompt_embeds], dim=0)

            vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)

            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, self.settings.vars["height"].value // vae_scale, self.settings.vars["width"].value // vae_scale),
                device="cuda",
                dtype=torch.float16,
            ) # todo: generator

            self.scheduler.set_timesteps(self.settings.vars["steps"].value)

            latents = latents * self.scheduler.init_noise_sigma

            original_size = (self.settings.vars["height"].value, self.settings.vars["width"].value)
            target_size = (self.settings.vars["height"].value, self.settings.vars["width"].value)
            crop_coords_top_left = (0, 0)

            add_time_ids = list(original_size + crop_coords_top_left + target_size)

            passed_add_embed_dim = (self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2.config.projection_dim)
            expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

            if passed_add_embed_dim != expected_add_embed_dim:
                print("embed dim is messed up")

            add_time_ids = torch.tensor([add_time_ids], dtype=torch.float16)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            add_time_ids = add_time_ids.to("cuda").repeat(1,1)

            pooled_embeds = torch.cat([pooled_negatives, pooled_positives], dim=0)

            added_cond_kwargs = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

            for step in self.scheduler.timesteps:
                latents_expanded = torch.cat([latents] * 2)
                latents_expanded = self.scheduler.scale_model_input(
                    latents_expanded, timestep=step
                )

                noise_prediction = self.unet(
                    latents_expanded, step, encoder_hidden_states=embeddings,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                (
                    noise_prediction_negative,
                    noise_prediction_positive,
                ) = noise_prediction.chunk(2)
                noise_prediction = noise_prediction_negative + self.settings.vars["scale"].value * (
                    noise_prediction_positive - noise_prediction_negative
                )

                latents = self.scheduler.step(
                    noise_prediction, step, latents
                ).prev_sample
            latents = 1 / self.vae.config.scaling_factor * latents
            latents = latents.float()
            images = self.vae.decode(latents).sample

            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")

            #flattened = [image.flatten().tobytes() for image in images]
            #base64ed = [base64.b64encode(image).decode("utf-8") for image in flattened]

            return [DiffusionResult(image) for image in images][0]

            # return {
            #     "schema": "image_bytes",
            #     "height": height,
            #     "width": width,
            #     "bytes": base64ed,
            # }
            #pil_images = [Image.fromarray(image) for image in images]
            #for p in pil_images:
            #    p.save("DIFFUSION_RESULT.png")


class HuggingFaceTextGenerator:
    def __init__(self, model: str):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
        config = transformers.AutoConfig.from_pretrained(model, trust_remote_code=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.tokenizer_name, low_cpu_mem_usage=True, dtype=torch.bfloat16
        )
        self.model_source = model
        self.tokenizer_source = config.tokenizer_name
        self.model.to(device="cuda")
        self.tokenizer
        self.limit = 100
        self.temperature = 0.8
        self.top_p = 0.945
        self.repetition_penalty = 5.0

    def sm_schematize(self, cache):
        return {"schema": "text", "value": "asdf **q**"}

    async def sm_inspect(self, content):
        return f"### 🤗 Text Generator\n\n**Model** `{self.model_source}`\n\n**Tokenizer** `{self.tokenizer_source}`\n\n**Temperature** `{self.temperature}`\n\n**Top-P** `{self.top_p}`\n\n**Repetition Penalty** `{self.repetition_penalty}`\n\n**Limit** `{self.limit}`"

    async def sm_limit(self, limit):
        try:
            limit = int(limit)
        except:
            return CommandFailure(f"Expected an integer, got {limit}")
        self.limit = limit

    async def sm_temp(self, temperature):
        try:
            temperature = float(temperature)
        except:
            return CommandFailure(f"Expected a number, got {temperature}")
        self.temperature = temperature

    async def sm_top_p(self, top_p):
        try:
            top_p = float(top_p)
        except:
            return CommandFailure(f"Expected a number, got {top_p}")
        self.top_p = top_p

    async def sm_default(self, content):
        return self.tokenizer.decode(
            self.model.generate(
                self.tokenizer(content, return_tensors="pt").input_ids.to("cuda"),
                temperature=self.temperature,
                max_length=self.limit,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )[0],
            skip_special_tokens=True,
        )

    async def sm_raw(self, content):
        return self.model.generate(
            self.tokenizer(content, return_tensors="pt").input_ids.to("cuda"),
            temperature=self.temperature,
            max_length=self.limit,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )[0]
