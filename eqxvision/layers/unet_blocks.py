from typing import Callable, List, Optional, Tuple, Union

import einops
import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float


def get_up_block(
    block_type: str,
):
    pass


def get_down_bloc(
    block_type: str,
):
    pass


class Upsample2D(eqx.Module):

    use_conv: bool
    conv: nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: Optional[bool] = False,
        padding: Optional[int] = 1,
        *,
        key: jrandom.PRNGKey = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        self.use_conv = use_conv
        out_channels = out_channels or in_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=padding,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, " in_channels height width"],
        output_size: Optional[Tuple[int, int]] = None,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Float[Array, " out_channels _ _"]:

        c, h, w = x.shape
        if output_size is None:
            x = jax.image.resize(x, shape=(c, h * 2, w * 2), method="nearest")
        else:
            x = jax.image.resize(
                x, shape=(c, output_size[0], output_size[1]), method="nearest"
            )
        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample2D(eqx.Module):

    use_conv: bool
    op: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_conv: Optional[bool] = True,
        padding: Optional[int] = 1,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        out_channels = out_channels or in_channels

        if use_conv:
            self.op = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
                key=key,
            )
        else:
            self.op = nn.AvgPool2D(
                kernel_size=2,
                stride=2,
            )

    def __call__(
        self,
        x: Float[Array, " in_channels height width"],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Float[Array, " out_channels _ _"]:

        return self.op(x)


class ResBlock2D(eqx.Module):

    time_embedding_proj: nn.Linear
    norm_in: nn.GroupNorm
    norm_out: nn.GroupNorm
    conv_in: nn.Conv2d
    conv_out: nn.Conv2d
    dropout: nn.Dropout
    conv_shortcut: eqx.Module
    upsample: Upsample2D
    downsample: Downsample2D
    activation_fn: Callable

    use_scale_shift_norm: bool
    output_scale_factor: float
    use_in_shortcut: bool
    up: bool
    down: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        embedding_channels: Optional[int] = None,
        groups_in: Optional[int] = 32,
        groups_out: Optional[int] = None,
        eps: Optional[float] = 1e-6,
        dropout: Optional[float] = 0.0,
        use_scale_shift_norm: Optional[bool] = False,
        output_scale_factor: Optional[float] = 1.0,
        use_in_shortcut: Optional[bool] = None,
        activation_fn=jax.nn.silu,
        up: Optional[bool] = False,
        down: Optional[bool] = False,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        """**Arguments:**

        - `in_channels`: Number of input channels.
        - `out_channels`: Number of output channels. If not provided, this will be `in_channels`.
        - `embedding_channels`: Number of dimension in time embedding.
        - `groups_in`: the number of groups of the input `self.norm_in`.
        - `groups_out`: the number of groups of the output `self.norm_out`. Default: `groups_in`.
        - `eps`: a parameter of `nn.GroupNorm`.
        - `dropout`: the dropout probability.
        - `use_scale_shift_norm`: whether to scale and shift of the output of `self.norm_out`.
        - `output_scale_factor`: scale the final output by a factor. Default: 1.0.
        - `use_in_shortcut`: whether to create skip connection.
        - `up`: whether to add upsample to input as well as hidden states after group norm
        - `down`: whether to add downsample to input as well as hidden states after group norm
        - `key`: Ignored

        **Returns:**

        Output of `eqx.nn.LayerNorm` applied to each `dim_0*dim_1 x c` entry.
        """

        if key is None:
            key = jrandom.PRNGKey(0)

        keys = jrandom.split(key, 4)
        out_channels = out_channels or in_channels

        self.norm_in = nn.GroupNorm(
            groups=groups_in,
            channels=in_channels,
            eps=eps,
        )
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=keys[0],
        )

        self.norm_out = nn.GroupNorm(
            groups=groups_out or groups_in,
            channels=out_channels,
            eps=eps,
        )

        self.conv_out = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            key=keys[1],
        )

        self.dropout = nn.Dropout(dropout)

        if embedding_channels is not None:
            embedding_out_dim = (
                2 * out_channels if use_scale_shift_norm else out_channels
            )
            self.time_embedding_proj = nn.Linear(
                embedding_channels,
                embedding_out_dim,
                key=keys[2],
            )
        else:
            self.time_embedding_proj = nn.Identity()

        if up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        else:
            self.upsample = nn.Identity()

        if down:
            self.downsample = Downsample2D(in_channels, use_conv=False)
        else:
            self.downsample = nn.Identity()

        # False if `use_in_shortcut` is not specified AND `in_channels` is not equal `out_channels`
        self.use_in_shortcut = (
            in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
        )
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                key=keys[3],
            )
        else:
            self.conv_shortcut = nn.Identity()

        self.activation_fn = activation_fn
        self.output_scale_factor = output_scale_factor
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up = up
        self.down = down

    def __call__(
        self,
        x: Float[Array, " in_channels height width"],
        time_embedding: Optional[Float[Array, " _"]] = None,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Float[Array, " out_channels _ _"]:

        h = self.norm_in(x)
        h = self.activation_fn(h)

        if self.up:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.down:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv_in(x)

        if time_embedding is not None:
            embedding_out = self.activation_fn(time_embedding)
            embedding_out = self.time_embedding_proj(embedding_out)
            embedding_out = jnp.reshape(
                embedding_out,
                embedding_out.shape + (1,) * (h.ndim - 1),
            )

            if self.use_scale_shift_norm:
                scale, shift = jnp.split(embedding_out, 2, axis=0)
                h = self.norm_out(h) * (1.0 + scale) + shift
            else:
                h = h + embedding_out
                h = self.norm_out(h)

        h = self.activation_fn(h)
        h = self.dropout(h, key=key)
        h = self.conv_out(h)

        if self.use_in_shortcut:
            x = self.conv_shortcut(x)

        output = (x + h) * self.output_scale_factor

        return output


class AttnBlock2D(eqx.Module):

    """
    Taken from
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L278
    which was originally from
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66
    """  # noqa

    norm: nn.GroupNorm
    qkv: nn.Conv1d
    proj_out: nn.Conv1d

    num_heads: int

    def __init__(
        self,
        in_channels: int,
        num_heads: Optional[int] = 1,
        num_head_channels: Optional[int] = -1,
        groups_in: Optional[int] = 32,
        eps: Optional[float] = 1e-6,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:
        if key is None:
            key = jrandom.PRNGKey(0)
        qkv_key, proj_key = jrandom.split(key)

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            self.num_heads == in_channels // num_head_channels

        self.norm = nn.GroupNorm(groups=groups_in, channels=in_channels)
        self.qkv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=3 * in_channels,
            kernel_size=1,
            key=qkv_key,
        )

        self.proj_out = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            key=proj_key,
        )

        # TODO: zero param project out

    def __call__(
        self,
        x: Float[Array, " in_channels height width"],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Float[Array, " in_channels height width"]:

        c, *spatial = x.shape
        x = jnp.reshape(x, (c, -1))
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = jnp.split(
            qkv,
            indices_or_sections=3,
            axis=0,
        )
        q, k, v = map(
            lambda x: einops.rearrange(x, "n (h d)-> h n d", h=self.num_heads),
            (q, k, v),
        )
        h = jax.vmap(nn.attention.dot_product_attention)(q, k, v)
        h = einops.rearrange(h, "h n d -> n (h d)", h=self.num_heads)
        h = self.proj_out(h)

        return (x + h).reshape(c, *spatial)


class CrossAttention(eqx.Module):

    scale: float
    heads: int

    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear

    out: eqx.Module

    def __init__(
        self,
        query_dim,
        context_dim: Optional[int] = None,
        heads: Optional[int] = 8,
        dim_heads: Optional[int] = 64,
        dropout: Optional[float] = 0.0,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ):
        if key is None:
            key = jrandom.PRNGKey(0)

        qkey, kkey, vkey, outkey = jrandom.split(key, 4)
        context_dim = context_dim if context_dim else query_dim
        inner_dim = dim_heads * heads
        self.heads = heads
        self.scale = dim_heads**-0.5

        self.query = eqx.nn.Linear(
            query_dim,
            inner_dim,
            use_bias=False,
            key=qkey,
        )
        self.key = eqx.nn.Linear(
            context_dim,
            inner_dim,
            use_bias=False,
            key=kkey,
        )
        self.value = eqx.nn.Linear(
            context_dim,
            inner_dim,
            use_bias=False,
            key=vkey,
        )

        self.out = eqx.nn.Sequential(
            layers=[
                eqx.nn.Linear(inner_dim, query_dim, key=outkey),
                eqx.nn.Dropout(dropout),
            ]
        )

    def __call__(
        self, x, context=None, mask=None, *, key: Optional[jrandom.PRNGKey] = None
    ):

        n_head = self.heads

        q = jax.vmap(self.query)(x)
        context = context if context is not None else x
        k = jax.vmap(self.key)(context)
        v = jax.vmap(self.value)(context)

        q, k, v = map(
            lambda x: self.scale * einops.rearrange(x, "n (h d) -> h n d", h=n_head),
            (q, k, v),
        )

        batch_dot_product = jax.vmap(eqx.nn.attention.dot_product_attention)
        if mask:
            mask = einops.rearrange(mask, "... -> (...)")
            mask = einops.repeat(mask, "j -> h j", h=n_head)
            mask = ~mask
            attn = batch_dot_product(q, k, v, mask=mask)
        else:
            attn = batch_dot_product(q, k, v)

        attn = einops.rearrange(attn, "h n d -> n (h d)", h=n_head)
        return jax.vmap(self.out)(attn, key=jrandom.split(key, attn.shape[0]))


class MidBlock2D(eqx.Module):

    resnets: List[eqx.Module]
    attentions: List[eqx.Module]

    def __init__(
        self,
        in_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        attn_num_head_channels: Optional[int] = 1,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        resnets = [
            ResBlock2D(
                channels=in_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
        ]
        key = jrandom.split(key, 1)[0]

        attentions = []
        for _ in range(num_layers):
            attn = AttnBlock2D(
                channels=in_channels,
                num_head_channels=attn_num_head_channels,
                use_new_attention_order=True,
                key=key,
            )
            attentions.append(attn)
            key = jrandom.split(key, 1)[0]

            resnet = ResBlock2D(
                channels=in_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            resnets.append(resnet)
            key = jrandom.split(key, 1)[0]

        self.attentions = attentions
        self.resnets = resnets

    def __call__(self, x, embedding=None, *, key):
        x = self.resnets[0](x, embedding, key=key)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x, key=key)
            x = resnet(x, embedding, key=key)

        return


class UpBlock2D(eqx.Module):

    resnets: List[ResBlock2D]
    upsamplers_0: Upsample2D

    add_upsample: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        add_upsample: Optional[bool] = True,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        self.add_upsample = add_upsample

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            res_in_channels = prev_out_channels if i == 0 else out_channels
            res_block = ResBlock2D(
                in_channels=res_skip_channels + res_in_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            resnets.append(res_block)
            key = jrandom.split(key, 1)[0]

        self.resnets = resnets
        if self.add_upsample:
            self.upsamplers_0 = Upsample2D(self.out_channels, key=key)
        else:
            self.upsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " channel height width"],
        residual: List[Float[Array, "..."]],
        time_embedding: Float[Array, "..."],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Tuple[Float[Array, "..."], List[Float[Array, "..."]]]:

        for resnet in self.resnets:
            res_hidden_states = residual.pop()
            x = jnp.concatenate([x, res_hidden_states], axis=0)
            x = resnet(x, time_embedding, key=key)

        if self.add_upsample:
            x = self.upsamplers_0(x)

        return x


class DownBlock2D(eqx.Module):

    resnets: List[ResBlock2D]
    downsamplers_0: Downsample2D

    add_downsample: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        add_downsample: Optional[bool] = True,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:
        if key is None:
            key = jrandom.PRNGKey(0)

        self.add_downsample = add_downsample

        resnets = []
        for i in range(num_layers):
            res_in_channels = in_channels if i == 0 else out_channels

            res_block = ResBlock2D(
                channels=res_in_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            resnets.append(res_block)
            key = jrandom.split(key, 1)[0]

        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = Downsample2D(self.out_channels, key=key)
        else:
            self.downsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " in_channels height width"],
        time_embedding: Float[Array, "..."],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Tuple[Array, List[Array]]:

        output_states = []

        for resnet in self.resnets:
            x = resnet(x, time_embedding, key=key)
            output_states += [x]

        if self.add_downsample:
            x = self.downsamplers_0(x)
            output_states += [x]

        return x, output_states


class AttnUpblock2D(eqx.Module):

    resnets: List[ResBlock2D]
    attentions: List[AttnBlock2D]
    upsamplers_0: Union[Upsample2D, nn.Identity]

    add_upsample: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        embedding_channels: int,
        num_layers: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        attn_num_head_channels: Optional[int] = 1,
        add_upsample: Optional[bool] = True,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        self.add_upsample = add_upsample
        resnets = []
        attentions = []

        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels

            res_in_channels = prev_out_channels if i == 0 else out_channels

            res_block = ResBlock2D(
                in_channels=res_in_channels + res_skip_channels,
                embedding_channels=embedding_channels,
                out_channels=out_channels,
                dropout=dropout,
                key=key,
            )

            key = jrandom.split(key, 1)[0]
            resnets.append(res_block)

            attn = AttnBlock2D(
                in_channels=out_channels,
                num_head_channels=attn_num_head_channels,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            attentions.append(attn)

        self.resnets = resnets
        self.attentions = attentions

        self.add_upsample = add_upsample
        if add_upsample:
            self.upsamplers_0 = Upsample2D(out_channels, key=key)
        else:
            self.upsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " channel height width"],
        residual: List[Float[Array, "..."]],
        time_embedding: Float[Array, "..."],
        *,
        key: jrandom.PRNGKey,
    ) -> Tuple[Array, List[Array]]:

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = residual.pop()
            x = jnp.concatenate([x, res_hidden_states], axis=0)
            x = resnet(x, time_embedding, key=key)
            x = attn(x, key=key)

        if self.add_upsample:
            x = self.upsamplers_0(x)

        return x


class AttnDownBlock2D(eqx.Module):

    resnets: List[ResBlock2D]
    attentions: List[AttnBlock2D]
    downsamplers_0: Union[Downsample2D, nn.Identity]

    add_downsample: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        dropout: Optional[float] = 0.0,
        num_layers: Optional[int] = 1,
        attn_num_head_channels: Optional[int] = 1,
        output_scale_factor: Optional[float] = 1.0,
        downsample_padding: Optional[int] = 1,
        add_downsample: Optional[bool] = True,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        self.add_downsample = add_downsample

        resnets, attentions = [], []

        for i in range(num_layers):

            res_in_channels = in_channels if i == 0 else out_channels
            resnet = ResBlock2D(
                in_channels=res_in_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            resnets.append(resnet)

            attn = AttnBlock2D(
                in_channels=out_channels,
                num_head_channels=attn_num_head_channels,
                key=key,
            )
            key = jrandom.split(key, 1)[0]
            attentions.append(attn)

        self.resnets = resnets
        self.attentions = attentions

        if add_downsample:
            self.downsamplers_0 = Downsample2D(
                res_in_channels,
                out_channels=out_channels,
                use_conv=True,
                key=key,
            )
        else:
            self.downsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " in_channel height width"],
        time_embedding: Float[Array, "..."],
        *,
        key: jrandom.PRNGKey = None,
    ) -> Tuple[Array, List[Array]]:

        output_states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, time_embedding, key=key)
            x = attn(x, key=key)
            output_states += [x]

        if self.add_downsample:
            x = self.downsamplers_0(x)
            output_states += [x]

        return x, output_states


class CrossAttnDownBlock(eqx.Module):
    add_downsample: bool

    resnets: List[eqx.Module]
    attentions: List[eqx.Module]
    downsamplers_0: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        attn_num_head_channels: int = 1,
        add_downsample: bool = True,
        *,
        key: jrandom.PRNGKey,
    ) -> None:

        self.add_downsample = add_downsample

        resnets = []
        attentions = []
        for i in range(self.num_layers):
            res_in_channels = in_channels if i == 0 else out_channels
            res_block = ResBlock2D(
                channels=res_in_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            resnets.append(res_block)
            key = jrandom.split(key, 1)[0]

            # this is the main different to normal DownBlock
            # TODO: SpatialTransformer
            # attn_block = SpatialTransformer(
            #     in_channels=out_channels,
            #     n_head=self.attn_num_head_channels,
            #     d_head=self.out_channels // self.attn_num_head_channels,
            #     key=key,
            # )
            # attentions.append(attn_block)
            key = jrandom.split(key, 1)[0]

        self.resnets = resnets
        self.attentions = attentions

        if add_downsample:
            self.downsamplers_0 = Downsample2D(out_channels, key=key)
        else:
            self.downsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " in_channel height width"],
        time_embedding: Float[Array, " embedding_channels"],
        encoder_hidden_states: Float[Array, "..."],
        *,
        key: jrandom.PRNGKey = None,
    ) -> Tuple[Array, List[Array]]:

        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, time_embedding, key=key)
            x = attn(x, encoder_hidden_states, key=key)
            output_states += [x]

        if self.add_downsample:
            x = self.downsamplers_0(x)
            output_states += [x]

        return x, output_states


class CrossAttnUpBlock2D(eqx.Module):

    resnets: List[eqx.Module]
    attentions: List[eqx.Module]
    upsamplers_0: eqx.Module

    add_upsample: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        embedding_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        attn_num_head_channels: int = 1,
        add_upsample: bool = True,
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> None:

        if key is None:
            key = jrandom.PRNGKey(0)

        self.add_upsample = add_upsample

        resnets = []
        attenions = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels

            res_in_channels = prev_out_channels if i == 0 else out_channels

            res_block = ResBlock2D(
                channels=res_in_channels + res_skip_channels,
                out_channels=out_channels,
                embedding_channels=embedding_channels,
                dropout=dropout,
                key=key,
            )
            resnets.append(res_block)
            key = jrandom.split(key, 1)[0]

            # TODO: SpatialTransformer
            # attn_block = SpatialTransformer(
            #     in_channels=out_channels,
            #     n_head=attn_num_head_channels,
            #     d_head=self.out_channels // attn_num_head_channels,
            #     key=key,
            # )
            # attenions.append(attn_block)
            key = jrandom.split(key, 1)[0]

        self.resnets = resnets
        self.attentions = attenions

        if self.add_upsample:
            self.upsamplers_0 = Upsample2D(out_channels, key=key)
        else:
            self.upsamplers_0 = eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, " n_channels height width"],
        residual: List[Float[Array, "..."]],
        time_embedding: Float[Array, "..."],
        encoder_hidden_states: Float[Array, "..."],
        *,
        key: Optional[jrandom.PRNGKey] = None,
    ) -> Tuple[Array, List[Array]]:

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = residual.pop()
            x = jnp.concatenate([x, res_hidden_states], axis=0)
            x = resnet(x, time_embedding, key=key)
            x = attn(x, encoder_hidden_states, key=key)

        if self.add_upsample:
            x = self.upsamplers_0(x)

        return x
