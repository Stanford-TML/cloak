import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.shared import masking as _masking

logger = logging.getLogger("openpi")

# Side-channel for reporting the actual masked token count from inside JIT
# to the outer inference loop. Written by jax.debug.callback in embed_prefix,
# read by Policy.infer after sample_actions returns.
last_masked_token_count: int | None = None


def _report_masked_token_count(count):
    global last_masked_token_count
    last_masked_token_count = int(count)


def _save_processed_images(images: dict, pixel_masks: dict | None = None, out_dir: str = "scratch/debug_model") -> None:
    """Host callback: write the preprocessed model images (in [-1, 1]), all
    cameras concatenated side-by-side into one PNG per batch element. When
    pixel_masks are present, also write a ``processed_mask_{i}.png`` with the
    mask overlaid in red on each camera that has one (same camera order)."""
    import pathlib

    import numpy as np
    from PIL import Image

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(images.keys())
    # [n_cam, b, h, w, c] in [-1, 1] -> uint8.
    arrs = np.stack([(((np.asarray(images[n]) + 1.0) * 127.5).clip(0, 255)).astype(np.uint8) for n in names])
    batch_size = arrs.shape[1]
    for i in range(batch_size):
        combined = np.concatenate([arrs[c, i] for c in range(len(names))], axis=1)  # along width
        Image.fromarray(combined).save(out / f"processed_{i}.png")

    if pixel_masks:
        for i in range(batch_size):
            panels = []
            for c, n in enumerate(names):
                panel = arrs[c, i].copy()
                if n in pixel_masks:
                    m = np.asarray(pixel_masks[n][i]).astype(bool)
                    panel[m] = (panel[m] * 0.5 + np.array([220, 30, 30], dtype=np.uint8) * 0.5).astype(np.uint8)
                panels.append(panel)
            Image.fromarray(np.concatenate(panels, axis=1)).save(out / f"processed_mask_{i}.png")

    logger.info(
        "Wrote %d concatenated image(s) [%s]%s to %s",
        batch_size, ", ".join(names),
        " + mask overlay" if pixel_masks else "", out,
    )


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        # Patch masking config for wrist cameras.
        self.patch_masking_strategy = config.patch_masking_strategy

        # Affordance token injection config.
        self.inject_affordance_tokens = config.inject_affordance_tokens
        if self.inject_affordance_tokens:
            # Per-finger learned embedding table (index 0=left, 1=right), added to the
            # patch posemb at each finger's patch.
            self.affordance_finger_emb = nnx.Embed(
                num_embeddings=2, features=paligemma_config.width, rngs=rngs
            )

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # Single source of truth for patch masks — same helper the sanity-check overlays call.
        patch_masks_by_name = _masking.compute_observation_patch_masks(
            obs.pixel_masks,
            obs.original_image_dims,
            obs.images.keys(),
            self.patch_masking_strategy,
            _model.IMAGE_RESOLUTION,
        )
        if (
            self.patch_masking_strategy is not None
            and obs.pixel_masks is None
            and any("wrist" in n for n in obs.images)
        ):
            logger.warning(
                "Patch masking strategy %r is configured but no pixel masks were provided. "
                "Proceeding without masking.",
                self.patch_masking_strategy,
            )
        # embed images
        for name in obs.images:
            siglip_patch_mask = None
            per_token_mask = None
            if name in patch_masks_by_name:
                patch_masks = patch_masks_by_name[name]  # [b, num_patches], True=masked
                siglip_patch_mask = ~patch_masks  # True=keep for SigLIP
                per_token_mask = siglip_patch_mask  # [b, num_patches], True=keep
                # Report the actual masked token count via side-channel for debug tooling.
                jax.debug.callback(_report_masked_token_count, patch_masks.sum())

            image_tokens, siglip_out = self.PaliGemma.img(
                obs.images[name], train=False, patch_mask=siglip_patch_mask
            )

            # Affordance token injection for the left wrist camera only. DROID uses a single
            # physical wrist cam (keyed `left_wrist_0_rgb`); `right_wrist_0_rgb` is a zero-padded
            # placeholder with image_mask=False, so injecting there is wasted work.
            if self.inject_affordance_tokens and name == "left_wrist_0_rgb":
                assert obs.affordance_pixels is not None, (
                    f"inject_affordance_tokens=True but no affordance_pixels provided for {name!r}. "
                    f"Ensure the dataset contains affordance data."
                )
                projected_posemb = siglip_out["projected_posemb"]  # [num_patches, emb]
                image_tokens, per_token_mask = self._inject_affordance_tokens(
                    image_tokens, per_token_mask, obs.affordance_pixels, projected_posemb
                )

            tokens.append(image_tokens)
            if per_token_mask is not None:
                # AND with camera-level mask so absent cameras are fully masked.
                camera_mask = einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
                input_mask.append(per_token_mask & camera_mask)
            else:
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[name],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _inject_affordance_tokens(
        self,
        image_tokens: jnp.ndarray,
        per_token_mask: jnp.ndarray | None,
        affordance_pixels: jnp.ndarray,
        projected_posemb: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Replace patch tokens at affordance locations with learned per-finger tokens.

        For each fingertip, the token is: projected_posemb[patch] + finger_emb.
        When both fingers map to the same patch, the embeddings are averaged:
            projected_posemb[patch] + (left_emb + right_emb) / 2

        Args:
            image_tokens: [b, num_patches, emb] — SigLIP output tokens.
            per_token_mask: [b, num_patches] bool (True=keep) or None.
            affordance_pixels: [b, 2, 2] — [[left_x, left_y], [right_x, right_y]].
            projected_posemb: [num_patches, emb] — SigLIP posemb projected through head.

        Returns:
            Updated (image_tokens, per_token_mask).
        """
        b = image_tokens.shape[0]
        patch_size = 14
        grid_size = 224 // patch_size  # 16

        # Compute patch centroids: [num_patches, 2] as (x, y).
        half = patch_size / 2.0
        cx = jnp.arange(grid_size) * patch_size + half
        cy = jnp.arange(grid_size) * patch_size + half
        grid_cx, grid_cy = jnp.meshgrid(cx, cy)
        centroids = jnp.stack([grid_cx.ravel(), grid_cy.ravel()], axis=-1)  # [256, 2]

        # Find nearest patch for each finger.
        left_coords = affordance_pixels[:, 0, :]   # [b, 2]
        right_coords = affordance_pixels[:, 1, :]  # [b, 2]

        def find_nearest(coords):
            dists = jnp.sum((coords[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)  # [b, 256]
            return jnp.argmin(dists, axis=-1)  # [b]

        left_idx = find_nearest(left_coords)    # [b] — flat patch indices
        right_idx = find_nearest(right_coords)  # [b]

        # Anchor: projected posemb at each finger's patch.
        left_posemb = projected_posemb[left_idx]    # [b, emb]
        right_posemb = projected_posemb[right_idx]  # [b, emb]

        # Per-finger learned embedding vectors (input-independent constants).
        left_emb = self.affordance_finger_emb(jnp.array(0, dtype=jnp.int32))   # [emb]
        right_emb = self.affordance_finger_emb(jnp.array(1, dtype=jnp.int32))  # [emb]

        token_dtype = image_tokens.dtype
        left_token = (left_posemb + left_emb[None, :]).astype(token_dtype)    # [b, emb]
        right_token = (right_posemb + right_emb[None, :]).astype(token_dtype)  # [b, emb]

        batch_idx = jnp.arange(b)
        same_patch = left_idx == right_idx  # [b]

        # When same patch: average the two finger embs → posemb + (left_emb + right_emb) / 2
        merged_token = (left_posemb + ((left_emb + right_emb) / 2.0)[None, :]).astype(token_dtype)  # [b, emb]

        # Two unconditional scatters; decide the value with where at the [b, emb] level.
        # When same_patch, both scatters write merged_token to the same index — idempotent.
        new_tokens = image_tokens.at[batch_idx, left_idx].set(
            jnp.where(same_patch[:, None], merged_token, left_token)
        )
        new_tokens = new_tokens.at[batch_idx, right_idx].set(
            jnp.where(same_patch[:, None], merged_token, right_token)
        )

        # Mark affordance patches as kept in per_token_mask.
        if per_token_mask is not None:
            per_token_mask = per_token_mask.at[batch_idx, left_idx].set(True)
            per_token_mask = per_token_mask.at[batch_idx, right_idx].set(True)

        return new_tokens, per_token_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        debug: bool = False,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if debug:
            jax.debug.callback(_save_processed_images, observation.images, observation.pixel_masks)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
