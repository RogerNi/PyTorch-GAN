import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import WeightedRandomSampler

import pickle
import matplotlib.pyplot as plt

SAVED_FOLDER = "images-sampled-on-prob"
os.makedirs(SAVED_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class SavedSamples():
    def __init__(self, total_n, device):
        self.device = device
        self.weights = torch.zeros(total_n, requires_grad=True, device=torch.device(device))
        self.samples = torch.zeros((total_n, 1, opt.img_size, opt.img_size), device=torch.device(device))
        self.num_samples = 0
        self.gen_weight = torch.tensor(1.0, requires_grad=True, device=torch.device(device))
        pass
    
    def add_samples(self, samples):
        n_samples = samples.shape[0]
        new_weight = self.gen_weight - torch.log(torch.tensor(n_samples)) # calculate the new weight for new added samples and the generator
        self.weights.data[self.num_samples: self.num_samples + n_samples] = new_weight # assign the new weight for new added samples
        self.gen_weight.data = new_weight # assign the new weight for the generator
        self.num_samples += n_samples
        self.samples[self.num_samples: self.num_samples + n_samples] = samples # save samples

    
    def get_samples(self, n):
        indices = list(WeightedRandomSampler(
            torch.nn.functional.softmax(self.weights[:self.num_samples], 0), n, replacement=False)) # weighted random samples
        
        # return samples, corresponding weights and the weight for the generator (weighted by n/self.num_samples)
        return self.samples[indices], self.weights[indices], self.gen_weight * n / self.num_samples 


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss(reduction='none') # set reduction to none to let it return one loss value per sample, not per batch

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# SavedSamples
saved_samples = SavedSamples(opt.n_epochs * dataloader.__len__() * opt.batch_size, "cuda" if torch.cuda.is_available() else "cpu")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Weights = torch.optim.Adam([saved_samples.weights, saved_samples.gen_weight], lr=opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

g_loss_list = []
d_real_loss_list = []
d_fake_loss_list = []

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        optimizer_Weights.zero_grad()


        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)
        
        if saved_samples.num_samples > 0:
            prev_gens, prev_weights, gen_weight = saved_samples.get_samples(imgs.shape[0]) # select saved samples (of the number of batch size)
            softmax_weights = torch.nn.functional.softmax(torch.cat((prev_weights, gen_weight.view(1))) , 0) # do softmax
            prev_pred = discriminator(prev_gens)
            
            # get weighted average, adversarial_loss return one loss value per sample, not per batch
            g_loss_prev = torch.sum(adversarial_loss(prev_pred, valid).squeeze() * softmax_weights[:-1]) 


            validity = discriminator(gen_imgs)
            g_loss = torch.mean(adversarial_loss(validity, valid)) * softmax_weights[-1] # get weighted loss
            
            if i % 100 == 0:
                print(torch.cat((prev_weights, gen_weight.view(1))))
                print(softmax_weights)

            g_loss += g_loss_prev
        else:
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs)
            g_loss = torch.mean(adversarial_loss(validity, valid))
        
        g_loss.backward(retain_graph=True) # set retain_graph to True to support two optimizers operating on one graph
        optimizer_G.step()
        optimizer_Weights.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss =  torch.mean(adversarial_loss(discriminator(real_imgs), valid))
        
        # Loss for fake images
        if saved_samples.num_samples > 0:
            d_fake_loss_prev = torch.sum(adversarial_loss(prev_pred, fake).squeeze() * softmax_weights[:-1])
            fake_pred = discriminator(gen_imgs.detach())
            d_fake_loss = torch.mean(adversarial_loss(fake_pred, fake)) * softmax_weights[-1]
            d_fake_loss += d_fake_loss_prev
        else:
            fake_pred = discriminator(gen_imgs.detach())
            d_fake_loss = torch.mean(adversarial_loss(fake_pred, fake))
        
        d_loss = (real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        
        # -------------------------
        #  Save generated samples
        # -------------------------
        
        saved_samples.add_samples(gen_imgs.detach())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D real loss: %f] [D fake loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), real_loss.item(), d_fake_loss.item(), g_loss.item())
        )
        
        g_loss_list.append(g_loss.item())
        d_real_loss_list.append(real_loss.item())
        d_fake_loss_list.append(d_fake_loss.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], SAVED_FOLDER + "/%d.png" % batches_done, nrow=5, normalize=True)

            
# save losses to file and draw a plot
with open('loss_lists', 'wb') as f:
    pickle.dump([g_loss_list, d_real_loss_list, d_fake_loss_list], f)
    
x_len = len(g_loss_list)
plt.plot(range(x_len), g_loss_list, label = "Generator loss")
plt.plot(range(x_len), d_real_loss_list, label = "Discriminator real loss")
plt.plot(range(x_len), d_fake_loss_list, label = "Discriminator fake loss")
plt.legend()
plt.xlabel('iter')
plt.ylabel('loss')
plt.savefig("loss_plot.svg", format="svg")