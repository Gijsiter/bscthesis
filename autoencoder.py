import torch
from torch import nn

from encoderblock import EncBlock
from mappingnet import MappingNet
from compressor import EncodingCompressor
from sirendecoder import SirenDecoder


class AutoEnc(nn.Module):
    def __init__(self,
                 n_in=64000,
                 n_out=64000,
                 enc_hidden=9,
                 enc_activ=nn.ReLU(),
                 max_channels=None,
                 comp_out=256,
                 dec_hid_ft=256,
                 dec_hidden=4,
                 dec_out=64000,
                 outermost_linear=True,
                 w_0=1500,
                 w_h=30,
                 map_hidden=[128, 128, 128],
                 mode="latent"
                 ):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Encoder parameters
        self.enc_hidden = enc_hidden
        self.enc_activ = enc_activ

        # Compressor
        self.max_channels = max_channels
        out_channels = 2**enc_hidden if max_channels is None else max_channels
        self.comp_in = out_channels * int(n_in / 2**enc_hidden)
        self.comp_out = comp_out

        # Decoder parameters
        self.dec_hid_ft = dec_hid_ft
        self.dec_hidden = dec_hidden
        self.dec_out = dec_out
        self.outermost_linear = outermost_linear
        self.w_0 = w_0
        self.w_h = w_h

        # Mapping parameters
        self.map_hidden = map_hidden

        self.mode = mode

        self.encoder = self.init_encoder()
        self.compressor = self.init_compressor()
        self.decoder = self.init_decoder()

        if self.mode == 'film':
            self.mapping_net = self.init_mapping_net()

    def init_encoder(self):
        blocks = []
        for i in range(self.enc_hidden):
            in_c, out_c = 2**i, 2**(i + 1)
            if self.max_channels:
                in_c = in_c if in_c < self.max_channels else self.max_channels
                out_c = out_c if out_c < self.max_channels else self.max_channels
            blocks.append(EncBlock(in_c, out_c, self.enc_activ))

        encoder = nn.Sequential(*blocks)

        return encoder

    def init_compressor(self):
        compressor = EncodingCompressor(self.comp_in, self.comp_out,
                                        self.enc_activ, self.mode)
        return compressor

    def init_decoder(self):
        if self.mode == 'latent':
            dec_in = self.comp_out
            dec_out = self.n_out
        elif self.mode == 'concat':
            dec_in = self.comp_out + 1
            dec_out = 1
        else:
            dec_in = 1
            dec_out = 1

        decoder = SirenDecoder(dec_in, self.dec_hid_ft, self.dec_hidden, dec_out,
                               first_omega_0=self.w_0, hidden_omega_0=self.w_h,
                               outermost_linear=self.outermost_linear)

        return decoder

    def init_mapping_net(self):
        map_in = self.comp_out

        mapping_nets = nn.ModuleList([
            MappingNet(map_in, self.map_hidden, nfeatures=self.dec_hid_ft)
            for _ in range(self.dec_hidden)
        ])

        return mapping_nets

    def get_mgrid(self, sidelen, dim=1):
        '''Generates a flattened grid of (x,y,...) coordinates in a range
        of -100 to 100.
        PARAMS:
        sidelen: int
        dim: int

        OUT:
        torch.Tensor
        '''
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)

        return mgrid

    def apply_gather(self, x, idx, device):
        idx = idx.to(device)
        x = x.squeeze(-1).gather(1, idx).unsqueeze(-1)

        return x

    def forward(self, input, idx, device='cpu'):
        k = len(input)
        enc = self.encoder(input)
        enc = torch.flatten(enc, start_dim=1)
        lat_code = self.compressor(enc)

        gamma = [1.]
        beta = [0.]

        if self.mode == 'concat':
            # Duplicate latent code, add timepoints and separate samples.
            timepoints = torch.tile(self.get_mgrid(self.dec_out), (k, 1)).to(device)
            timepoints = timepoints.reshape(k, -1, 1)

            if idx is not None:
                timepoints = self.apply_gather(timepoints, idx, device)

            con_shape = torch.LongTensor([timepoints.shape[1]]).to(device)
            lat_code = torch.repeat_interleave(lat_code, con_shape, dim=0)
            lat_code = lat_code.reshape(k, -1, self.comp_out)
            lat_code = torch.cat((timepoints, lat_code), -1)

        elif self.mode == 'film':
            gambet = [net(lat_code) for net in self.mapping_net]
            gamma = torch.hstack([pair[0] for pair in gambet])
            beta = torch.hstack([pair[1] for pair in gambet])

            gamma = gamma.reshape(k, self.dec_hidden, -1)
            beta = beta.reshape(k, self.dec_hidden, -1)

            lat_code = self.get_mgrid(self.dec_out).repeat(k, 1, 1).to(device)
            lat_code = lat_code.reshape(k, -1, 1)

            if idx is not None:
                lat_code = self.apply_gather(lat_code, idx, device)

        out = self.decoder(lat_code, gamma, beta, self.mode)

        return out.reshape(k, 1, -1)
