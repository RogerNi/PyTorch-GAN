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

IMAGE_SAVE_FOLDER = "images-2"

os.makedirs(IMAGE_SAVE_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
# parser.add_argument("--n_samples", type=int, default=1000, help="samples to save per epoch")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

class SavedSamples():
    def __init__(self, total_n, device):
        self.weights = torch.zeros(total_n, requires_grad=True, device=torch.device(device))
        self.samples = None
        self.labels = None
        self.num_samples = 0
        self.gen_weight = torch.tensor(1.0, requires_grad=True, device=torch.device(device))
        pass
    
    def add_samples(self, samples, labels):
        n_samples = samples.shape[0]
        new_weight = self.gen_weight - torch.log(torch.tensor(n_samples))
        self.weights.data[self.num_samples: self.num_samples + n_samples] = new_weight
        # print(n_samples, new_weight, self.gen_weight)
        self.gen_weight.data = new_weight
        # print(self.gen_weight)
        self.num_samples += n_samples
        
        if self.samples is None:
            self.samples = samples
            self.labels = labels
        else:
            self.samples = torch.cat((self.samples, samples))
            self.labels = torch.cat((self.labels, labels))

    
    def get_samples(self, n):
        indices = torch.randperm(self.num_samples)[:n]
        return self.samples[indices], self.labels[indices], self.weights[indices], self.gen_weight * n / self.num_samples
    
    
    


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss(reduction='none')
auxiliary_loss = torch.nn.CrossEntropyLoss(reduction='none')

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

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

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor




def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, IMAGE_SAVE_FOLDER + "/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        optimizer_Weights.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)
        
        if saved_samples.num_samples > 0:
            prev_gens, prev_labels, prev_weights, gen_weight = saved_samples.get_samples(batch_size)
            softmax_weights = torch.nn.functional.softmax(torch.cat((prev_weights, gen_weight.view(1))) , 0)
            prev_pred, prev_aux = discriminator(prev_gens)
            g_loss_prev = torch.sum((adversarial_loss(prev_pred, valid).squeeze() + auxiliary_loss(prev_aux, prev_labels).squeeze()) / 2 * softmax_weights[:-1])
            # print(adversarial_loss(prev_pred, fake).squeeze(), softmax_weights[:-1])


            validity, pred_label = discriminator(gen_imgs)
            g_loss = torch.mean(0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))) * softmax_weights[-1]
            
            if i % 100 == 0:
                print(torch.cat((prev_weights, gen_weight.view(1))))
                print(softmax_weights)

            g_loss += g_loss_prev
        else:
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = torch.mean(0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)))

        g_loss.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_Weights.step()


        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()


        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = torch.mean((adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2)

        # Loss for fake images
        if saved_samples.num_samples > 0:
            d_fake_loss_prev = torch.sum((adversarial_loss(prev_pred, fake).squeeze() + auxiliary_loss(prev_aux, prev_labels).squeeze()) / 2 * softmax_weights[:-1])
            # print(adversarial_loss(prev_pred, fake).squeeze(), softmax_weights[:-1])
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = torch.mean(0.5 * (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels))) * softmax_weights[-1]
            d_fake_loss += d_fake_loss_prev
        else:
            fake_pred, fake_aux = discriminator(gen_imgs.detach())
            d_fake_loss = torch.mean((adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()
        
        # -------------------------
        #  Save generated samples
        # -------------------------
        
        saved_samples.add_samples(gen_imgs.detach(), gen_labels.detach())
        

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
            
        # if i > 40:
        #     exit()
