"""SeedVR2-7B MLX-native upscaler.

Loads the pre-converted MLX weights shipped in a benc0 model directory and runs the
one-step SeedVR2 super-resolution pipeline, reusing mflux building blocks (transformer,
VAE, scheduler, latent/condition creators, image utils).

Works with all benc0 SeedVR2-7B repos, selected purely by the local weights directory:
  - benc0/SeedVR2-7B-mlx            (fp16)
  - benc0/SeedVR2-7B-mlx-int8       (int8, ~9.3 GB)
  - benc0/SeedVR2-7B-sharp-mlx      (fp16, sharp variant)
  - benc0/SeedVR2-7B-sharp-mlx-int8 (int8, sharp variant)

The int8 vs fp16 difference is detected automatically from ``config.json`` (presence of
a ``"quantization"`` field): int8 repos quantize the transformer to 8-bit before loading,
fp16 repos load the transformer weights directly. The VAE and text embedding are always
fp16 in every repo, so they load identically.

A model directory must contain (produced by ``huggingface_hub.snapshot_download``):
  config.json, transformer.safetensors, vae.safetensors, pos_emb.safetensors
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.model.seedvr2_text_encoder.text_embeddings import SeedVR2TextEmbeddings
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition
from mflux.utils.image_util import ImageUtil
from mflux.utils.metadata_reader import MetadataReader
from mflux.utils.scale_factor import ScaleFactor


class SeedVR2MLX:
    """MLX-native SeedVR2-7B upscaler backed by a local pre-converted weights directory."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        cfg = json.loads((self.model_dir / "config.json").read_text())

        # --- Transformer (precision-agnostic) -----------------------------------
        self.transformer = SeedVR2Transformer(**cfg["transformer_overrides"])
        if "quantization" in cfg:
            q = cfg["quantization"]
            nn.quantize(
                self.transformer,
                group_size=q["group_size"],
                bits=q["bits"],
                class_predicate=SeedVR2WeightDefinition.quantization_predicate,
            )
            self.bits = int(q["bits"])
        else:
            self.bits = 16
        tx = mx.load(self.model_dir / "transformer.safetensors")
        self.transformer.update(tree_unflatten(list(tx.items())))
        mx.eval(self.transformer.parameters())

        # --- VAE (always fp16) ---------------------------------------------------
        self.vae = SeedVR2VAE()
        vae = mx.load(self.model_dir / "vae.safetensors")
        self.vae.update(tree_unflatten(list(vae.items())))
        mx.eval(self.vae.parameters())

        # --- Text conditioning (fixed prompt) ------------------------------------
        # Prefer the repo's own pos_emb so we use exactly what the weights were
        # validated against; fall back to mflux's bundled copy if absent.
        pos_emb_file = self.model_dir / "pos_emb.safetensors"
        if pos_emb_file.exists():
            emb = mx.load(str(pos_emb_file))["embedding"]
            if emb.ndim == 2:
                emb = emb[None, ...]
        else:
            emb = SeedVR2TextEmbeddings.load_positive()
        self.txt_pos = emb

        # --- Scaffolding (mirrors mflux SeedVR2Initializer._init_config) ---------
        self.model_config = ModelConfig.seedvr2_7b()
        self.callbacks = CallbackRegistry()
        self.tiling_config = TilingConfig()

    def upscale(self, image_path, resolution, softness=0.0, input_noise_scale=0.0, seed=42, tiling_config=None):
        """Run the one-step SeedVR2 upscale.

        Args:
            image_path: path to the input image (any format PIL can open).
            resolution: target shortest-side size in pixels (int) OR a scale-factor
                string such as ``"2x"`` / ``"3x"``.
            softness: 0.0-1.0 pre-downsampling control (0 disables; 1.0 -> up to 8x).
            seed: random seed for the noise latents.
            tiling_config: optional TilingConfig override for large-image memory control.

        Returns:
            mflux GeneratedImage (has a ``.image`` PIL.Image and a ``.save()`` method).
        """
        tiling_config = tiling_config or self.tiling_config
        image_path = Path(image_path)
        if isinstance(resolution, str):
            resolution = ScaleFactor.parse(resolution)

        # Pin to the GPU stream explicitly. The dedicated MLX thread (set up in app.py)
        # already has the stream, but this is a safety net if called from elsewhere.
        with mx.stream(mx.Device(mx.gpu)):
            # 0. Process and scale the input image (-> [-1,1] CHW batched, padded to /16)
            processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
                image_path=image_path,
                resolution=resolution,
                softness=softness,
            )

            # 1. One-step config (SeedVR2 is a single-step, guidance-free restoration model)
            config = Config(
                width=true_width,
                height=true_height,
                guidance=1.0,
                num_inference_steps=1,
                image_path=image_path,
                scheduler="seedvr2_euler",
                model_config=self.model_config,
            )

            # 2. Initial latents + static condition (encoded latent concatenated with a mask)
            initial_latent = VAEUtil.encode(vae=self.vae, image=processed_image, tiling_config=tiling_config)
            if input_noise_scale > 0.0:
                noise = mx.random.normal(
                    shape=initial_latent.shape,
                    dtype=initial_latent.dtype,
                ) * input_noise_scale
                initial_latent = initial_latent + noise
            static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
            latents = SeedVR2LatentCreator.create_noise_latents(
                seed=seed,
                height=initial_latent.shape[-2],
                width=initial_latent.shape[-1],
            )
            # Give the VAE encoder's (potentially large) activation buffers back to the OS now that
            # we only need the compact latent. processed_image is kept for color correction.
            mx.clear_cache()

            # 3-7. Single denoise step
            ctx = self.callbacks.start(seed=seed, prompt="", config=config)
            ctx.before_loop(latents)
            for t in config.time_steps:
                model_input = mx.concatenate([latents, static_condition], axis=1)
                noise = self.transformer(
                    txt=self.txt_pos,
                    vid=model_input,
                    timestep=config.scheduler.timesteps[t],
                )
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            ctx.after_loop(latents)

            # 8-9. Decode, crop to the true (unpadded) size, and color-correct against the style
            decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=tiling_config)
            # The decoder's upsample feature maps are the largest transient; return them to the OS
            # before we crop / color-correct (the final image is small in comparison).
            mx.clear_cache()
            decoded = decoded[:, :, :true_height, :true_width]
            style = processed_image[:, :, :true_height, :true_width]
            decoded = SeedVR2Util.apply_color_correction(decoded, style)

            init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None
            return ImageUtil.to_image(
                seed=seed,
                prompt="",
                config=config,
                quantization=self.bits,
                decoded_latents=decoded,
                generation_time=config.time_steps.format_dict["elapsed"],
                init_metadata=init_metadata,
            )
