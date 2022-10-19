import jax.random as jrandom

from eqxvision.layers.unet_blocks import AttnBlock2D, ResBlock2D


class TestUNetBlock:
    def test_resblock(self, getkey):

        in_channels = 32
        out_channels = 32
        image_size = 28
        embedding_channels = 10

        resblock = ResBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_channels=embedding_channels,
            key=getkey(),
        )

        x = jrandom.normal(key=getkey(), shape=(in_channels, image_size, image_size))
        time_embedding = jrandom.normal(key=getkey(), shape=(embedding_channels,))
        resblock(x, time_embedding, key=getkey())

    def test_attnblock(self, getkey):

        in_channels = 32
        image_size = 28

        attn = AttnBlock2D(in_channels=in_channels, key=getkey())

        x = jrandom.normal(key=getkey(), shape=(in_channels, image_size, image_size))
        attn(x, key=getkey())

    def test_upblock(self):
        pass

    def test_downblock(self):
        pass

    def test_midblock(self):
        pass


class TestUNetModel:
    def test_model(self):
        pass
