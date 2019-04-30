import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbearer as tb
import state_keys as keys


class BaseTempModel(nn.Module):
    def __init__(self):
        super(BaseTempModel, self).__init__()

    def encode_shape(self, x):
        ...

    def encode_motion(self, x):
        ...

    def decode(self, z):
        ...

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def eval_img(self, x, v):
        mu, logvar = self.encode_shape(x)
        z = mu
        tangle = torch.cat([z, v], 1)
        return self.decode(tangle).view(x.shape)

    def forward(self, x, state):
        F1, F2 = x, state[tb.Y_TRUE]

        mu, logvar = self.encode_shape(F1)
        z = self.reparameterize(mu, logvar)

        in_motion = torch.cat([F1, F2], 1)
        muv, logvarv = self.encode_motion(in_motion)
        v = self.reparameterize(muv, logvarv)

        tangle = torch.cat([z, v], 1)
        state[keys.MU], state[keys.LOGVAR] = mu, logvar
        state[keys.MUV], state[keys.LOGVARV] = muv, logvarv
        state[keys.V], state[keys.Z] = v, z
        state[keys.COMB] = torch.cat([mu, muv], 1)

        return self.decode(tangle).view(x.shape)


class MLP_Model(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=1):
        super(MLP_Model, self).__init__()

        # Shape
        self.fc1 = nn.Linear(28*28*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)

        # Motion
        self.fc1_m = nn.Linear(n_channel*28*28*2, 784)
        self.fc2_m = nn.Linear(784, 400)
        self.fc3_m = nn.Linear(400, 256)
        self.fc4_m = nn.Linear(256, n_motion * 2)

        # Decoder
        self.fc3 = nn.Linear(n_shape + n_motion, 800)
        self.fc4 = nn.Linear(800, 28*28*n_channel)

    def encode_shape(self, x):
        h1 = F.relu(self.fc1(x.view(x.shape[0], -1)))
        return self.fc21(h1), self.fc22(h1)

    def encode_motion(self, x):
        x = F.relu(self.fc1_m(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2_m(x))
        x = F.relu(self.fc3_m(x))
        x = self.fc4_m(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class Double_MLP_Model(BaseTempModel):
    def __init__(self, n_feat=2, n_shape=80, n_channel=1):
        super(Double_MLP_Model, self).__init__()

        # Shape
        self.fc1 = nn.Linear(28*28*2*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)

        # Motion
        self.fc1_m = nn.Linear(2*n_channel*28*28*2, 784)
        self.fc2_m = nn.Linear(784, 400)
        self.fc3_m = nn.Linear(400, 256)
        self.fc4_m = nn.Linear(256, n_feat * 2)

        # Decode
        self.fc3 = nn.Linear(n_shape+n_feat, 800)
        self.fc4 = nn.Linear(800, 28*28*2*n_channel)

    def encode_shape(self, x):
        h1 = F.relu(self.fc1(x.view(x.shape[0], -1)))
        return self.fc21(h1), self.fc22(h1)

    def encode_motion(self, x):
        x = F.relu(self.fc1_m(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2_m(x))
        x = F.relu(self.fc3_m(x))
        x = self.fc4_m(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class Shapes3dModel(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=1):
        super().__init__()
        self.shape_encoder = nn.Sequential(
            nn.Conv2d(n_channel, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
        )
        self.shape_fc = nn.Linear(64*4*4, 256)
        self.shape_fc2 = nn.Linear(256, 2*n_shape)

        self.motion_encoder = nn.Sequential(
            nn.Conv2d(n_channel*2, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
        )
        self.motion_fc = nn.Linear(64*4*4, 256)
        self.motion_fc2 = nn.Linear(256, 2*n_motion)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, n_channel, 4, 2, 1),
        )
        self.decoder_fc1 = nn.Linear(n_shape+n_motion, 256)
        self.decoder_fc2 = nn.Linear(256, 4*4*64)

    def encode_shape(self, x):
        x = self.shape_encoder(x)
        x = self.shape_fc(x.view(x.shape[0], -1))
        x =  self.shape_fc2(x)
        mu, logvar = x[:, :x.shape[-1]//2], x[:, x.shape[-1]//2:]
        return mu, logvar

    def encode_motion(self, x):
        x = self.motion_encoder(x)
        x = self.motion_fc(x.view(x.shape[0], -1))
        x = self.motion_fc2(x)
        mu, logvar = x[:, :x.shape[-1] // 2], x[:, x.shape[-1] // 2:]
        return mu, logvar

    def decode(self, z):
        z = self.decoder_fc1(z).relu()
        z = self.decoder_fc2(z)
        z = self.decoder(z.view(z.shape[0], 64, 4, 4))
        return z.sigmoid()


class dSprites_MLP_Model(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=1):
        super(dSprites_MLP_Model, self).__init__()

        # Shape
        self.fc1 = nn.Linear(64*64*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)

        # Motion
        self.fc1_m = nn.Linear(n_channel*64*64*2, 784)
        self.fc2_m = nn.Linear(784, 400)
        self.fc3_m = nn.Linear(400, 256)
        self.fc4_m = nn.Linear(256, n_motion * 2)

        # Decoder
        self.fc3 = nn.Linear(n_shape + n_motion, 800)
        self.fc4 = nn.Linear(800, 64*64*n_channel)

    def encode_shape(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode_motion(self, x):
        x = F.relu(self.fc1_m(x))
        x = F.relu(self.fc2_m(x))
        x = F.relu(self.fc3_m(x))
        x = self.fc4_m(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class chairs_MLP_Model(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=1):
        super(chairs_MLP_Model, self).__init__()

        # Shape
        self.fc1 = nn.Linear(64*64*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)

        # Motion
        self.fc1_m = nn.Linear(n_channel*64*64*2, 784)
        self.fc2_m = nn.Linear(784, 400)
        self.fc3_m = nn.Linear(400, 256)
        self.fc4_m = nn.Linear(256, n_motion * 2)

        # Decoder
        self.fc3 = nn.Linear(n_shape + n_motion, 800)
        self.fc4 = nn.Linear(800, 64*64*n_channel)

    def encode_shape(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode_motion(self, x):
        x = F.relu(self.fc1_m(x))
        x = F.relu(self.fc2_m(x))
        x = F.relu(self.fc3_m(x))
        x = self.fc4_m(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class ChairsCNNModel(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=1):
        super(ChairsCNNModel, self).__init__()

        # Shape
        self.shape_encoder = nn.Sequential(         # 512x512
            nn.Conv2d(n_channel, 32, 4, 2, 1),      # 256x256
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # 128x128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),             # 64x64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),            # 32x32
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),           # 16x16
            nn.ReLU(True),
        )
        self.shape_mu_logvar = nn.Linear(16*16*128, n_shape*2)
        self.shape_up = nn.Linear(n_shape, 16*16*128)


        # Motion
        self.motion_encoder = nn.Sequential(        # 512x512
            nn.Conv2d(n_channel*2, 32, 4, 2, 1),    # 256x256
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),             # 128x128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),             # 64x64
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # 32x32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # 16x16
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # 8x8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),             # 4x4
            nn.ReLU(True),
        )
        self.motion_mu_logvar = nn.Linear(4*4*64, n_motion*2)
        self.motion_up = nn.Linear(n_motion, 64*16*16)


        # Decoder
        self.decoder = nn.Sequential(                       # 16x16
            nn.ConvTranspose2d(192, 128, 4, 2, 1),          # 32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),           # 64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),            # 128x128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),            # 256x256
            nn.ReLU(True),
            nn.ConvTranspose2d(32, n_channel, 4, 2, 1),     # 512x512
            nn.Sigmoid()
        )

    def encode_shape(self, x):
        x =  self.shape_encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.shape_mu_logvar(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def encode_motion(self, x):
        x = self.motion_encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.motion_mu_logvar(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def eval_img(self, x, v):
        mu, logvar = self.encode_shape(x)
        z = mu
        z = self.shape_up(z)
        v = self.motion_up(v)
        tangle = torch.cat([z, v], 1).view(z.shape[0], -1, 16, 16)
        return self.decode(tangle).view(x.shape)

    def forward(self, x, state):
        F1, F2 = x, state[tb.Y_TRUE]

        mu, logvar = self.encode_shape(F1)
        z = self.reparameterize(mu, logvar)
        z = self.shape_up(z).view(mu.shape[0], -1, 16, 16)

        in_motion = torch.cat([F1, F2], 1)
        muv, logvarv = self.encode_motion(in_motion)
        v = self.reparameterize(muv, logvarv)
        v = self.motion_up(v).view(muv.shape[0], -1, 16, 16)

        tangle = torch.cat([z, v], 1)
        state[keys.MU], state[keys.LOGVAR] = mu, logvar
        state[keys.MUV], state[keys.LOGVARV] = muv, logvarv
        state[keys.V], state[keys.Z] = v, z

        return self.decode(tangle)


class VanillaVAE(torch.nn.Module):
    def __init__(self, n_shape=80, n_channel=1):
        super(VanillaVAE, self).__init__()

        self.fc1 = nn.Linear(28*28*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)
        self.fc3 = nn.Linear(n_shape, 800)
        self.fc4 = nn.Linear(800, 28*28*n_channel)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def eval_img(self, x, state):
        mu, logvar = self.encode(x.view(x.shape[0], -1))
        z = mu
        return self.decode(z).view(x.shape)

    def forward(self, x, state):
        mu, logvar = self.encode(x.view(x.shape[0], -1))
        z = self.reparameterize(mu, logvar)

        state[keys.MU], state[keys.LOGVAR] = mu, logvar
        state[keys.Z] = z

        return self.decode(z).view(x.shape)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


# Model with separate colour and motion spaces
class MLP_Model_Colour(BaseTempModel):
    def __init__(self, n_motion=2, n_colour=2, n_shape=80, n_channel=1):
        super(MLP_Model_Colour, self).__init__()

        # Shape
        self.fc1 = nn.Linear(28*28*n_channel, 800)
        self.fc21 = nn.Linear(800, n_shape)
        self.fc22 = nn.Linear(800, n_shape)

        # Motion
        self.fc1_m = nn.Linear(1*28*28*2, 784)
        self.fc2_m = nn.Linear(784, 400)
        self.fc3_m = nn.Linear(400, 256)
        self.fc4_m = nn.Linear(256, n_motion * 2)

        # Colour
        self.fc1_c = nn.Linear(3*28*28*1, 784)
        self.fc2_c = nn.Linear(784, 400)
        self.fc3_c = nn.Linear(400, 256)
        self.fc4_c = nn.Linear(256, n_colour * 2)

        # Decoder
        self.fc3 = nn.Linear(n_shape + n_motion + n_colour, 800)
        self.fc4 = nn.Linear(800, 28*28*n_channel)

    def encode_shape(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode_motion(self, x):
        x = F.relu(self.fc1_m(x))
        x = F.relu(self.fc2_m(x))
        x = F.relu(self.fc3_m(x))
        x = self.fc4_m(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def encode_colour(self, x):
        x = F.relu(self.fc1_c(x))
        x = F.relu(self.fc2_c(x))
        x = F.relu(self.fc3_c(x))
        x = self.fc4_c(x)
        mu, logvar = x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def eval_img(self, x, v, c):
        mu, logvar = self.encode_shape(x.view(x.shape[0], -1))
        z = mu
        tangle = torch.cat([z, v, c], 1)
        return self.decode(tangle).view(x.shape)

    def forward(self, x, state):
        mu, logvar = self.encode_shape(x.view(x.shape[0], -1))
        z = self.reparameterize(mu, logvar)

        moved = torch.cat([torch.mean(x, dim=1, keepdim=True), state[keys.MOVED]], 1)
        muv, logvarv = self.encode_motion(moved.view(x.shape[0], -1))
        v = self.reparameterize(muv, logvarv)

        coloured = state[tb.Y_TRUE]
        muc, logvarc = self.encode_colour(coloured.view(x.shape[0], -1))
        c = self.reparameterize(muc, logvarc)

        tangle = torch.cat([z, v, c], 1)
        state[keys.MU], state[keys.LOGVAR] = mu, logvar
        state[keys.MUV], state[keys.LOGVARV] = muv, logvarv
        state[keys.MUC], state[keys.LOGVARC] = muc, logvarc
        state[keys.V], state[keys.Z], state[keys.C] = v, z, c

        return self.decode(tangle).view(x.shape)


class UcfModel(BaseTempModel):
    def __init__(self, n_motion=2, n_shape=80, n_channel=3):
        super(UcfModel, self).__init__()

        # Shape
        self.shape_encoder = torch.nn.Sequential(
            nn.Conv2d(n_channel, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fcs = nn.Linear(15*20*128, n_shape*2)

        # Motion
        self.motion_encoder = torch.nn.Sequential(
            nn.Conv2d(n_channel*2, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
        )
        self.fcm = nn.Linear(15*20*128, n_motion*2)

        # Decoder
        self.fcd = nn.Linear(n_motion+n_shape, 15*20*128)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, n_channel, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode_shape(self, x):
        h1 = self.shape_encoder(x)
        h2 = self.fcs(h1.view(x.shape[0], -1))
        mu, logvar = h2[:, :h2.shape[1]//2], h2[:, h2.shape[1]//2:]
        return mu, logvar

    def encode_motion(self, x):
        h1 = self.motion_encoder(x)
        h2 = self.fcm(h1.view(x.shape[0], -1))
        mu, logvar = h2[:, :h2.shape[1]//2], h2[:, h2.shape[1]//2:]
        return mu, logvar

    def decode(self, z):
        h1 = self.fcd(z)
        return self.decoder(h1.view(z.shape[0], 128, 15, 20))

