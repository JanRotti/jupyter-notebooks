class Generator(nn.Module):
    def __init__(self, latent_dim = 100, label_dim = 10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.conditioned = latent_dim + label_dim
        self.net = nn.Sequential(
                nn.ConvTranspose2d(self.conditioned, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False),
                nn.Tanh()
                )
        
    def forward(self, input, labels):
        input = torch.cat((input,labels),1)
        output = input.unsqueeze(2).unsqueeze(3)
        return self.net(output)


class Discriminator(nn.Module):
    def __init__(self, label_dim = 10):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1, bias = False),
                nn.LayerNorm([64,16,16]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.LayerNorm([128,8,8]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.LayerNorm([256,4,4]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.LayerNorm([512,2,2]),
                nn.LeakyReLU(0.2),
                )
        self.OutputCON = nn.Conv2d(512, label_dim, 2, 1)
        self.OutputREG = nn.Conv2d(512, 1, 2, 1)
    
    def forward(self, input):
        net = self.net(input)
        Labels = self.OutputCON(net).squeeze()
        Output = self.OutputREG(net).view(-1)
        return Output, Labels
