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

from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler

import pickle
import matplotlib.pyplot as plt

from tqdm import tqdm

import time
import json

#===========Redirect message to tqdm================
import inspect
# store builtin print
old_print = print
def new_print(*args, **kwargs):
    # if tqdm.tqdm.write raises error, use builtin print
    try:
        tqdm.write(*args, **kwargs)
    except:
        old_print(*args, ** kwargs)
# globaly replace print with new_print
inspect.builtins.print = new_print
#====================================================

curr_time = time.strftime("%Y%m%d-%H%M%S")
SAVED_FOLDER = curr_time + "/images-sampled-on-prob"
os.makedirs(SAVED_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--weight_lr", type=float, default=0.02, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--sample_saving_delay", type=int, default=0, help="number of epochs to delay before starting to save samples")
parser.add_argument("--sample_saving_freq", type=int, default=500, help="frequency (iterations) to save samples")
parser.add_argument("--new_weight_ratio", type=float, default=0.5, help="The ratio of the new weights for samples to the weight for generator. Default: 0.5, new sample weights each have the same value as the new weight for the generator. If set to 0, the new sample weights are 0 while the generator weight remains unchanged.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="A small value to avoid weight from becoming 0")
parser.add_argument("--skip_weights", default=False, action="store_true", help="Whether to skip optimizing weights")
parser.add_argument("--disable_gpu", default=False, action="store_true", help="Whether to disable GPU")
parser.add_argument("--min_gen_weight", type=float, default=torch.finfo().min, help="the lower bound of raw sampling weight of the generator")
parser.add_argument("--min_gen_norm_weight", type=float, default=0, help="the lower bound of normalized sampling weight of the generator, valid range: [0, 1]")
parser.add_argument("--policy_loss", default=False, help="whether to use policy gradient loss instead of binary cross entropy loss")


opt = parser.parse_args()
print(opt)
with open(curr_time + '/commandline_args.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() and not opt.disable_gpu else False


class SavedSamples():
    def __init__(self, total_n, device):
        self.device = device
        self.weights = torch.ones(total_n + 1, requires_grad=True, device=torch.device(device)) # the first (0th) weight is for the generator
        self.unif = torch.ones(total_n + 1)
        self.samples = torch.zeros((total_n + 1, 1, opt.img_size, opt.img_size))
        self.num_samples = 1
        pass
    
    def add_samples(self, samples):
        n_samples = samples.shape[0]
        # new_weight = self.weights.data[0] - torch.log(torch.tensor(n_samples + 1)) # calculate the new weight for new added samples and the generator
        sample_ratio = (opt.new_weight_ratio + (opt.epsilon if opt.new_weight_ratio == 0 else 0)) / (opt.new_weight_ratio * n_samples + 1 - opt.new_weight_ratio)
        gen_ratio = (1 - opt.new_weight_ratio + (opt.epsilon if opt.new_weight_ratio == 1 else 0)) / (opt.new_weight_ratio * n_samples + 1 - opt.new_weight_ratio)
        new_gen_weight = self.weights.data[0] + torch.log(torch.tensor(gen_ratio))
        new_sample_weight = self.weights.data[0] + torch.log(torch.tensor(sample_ratio))
        self.weights.data[self.num_samples: self.num_samples + n_samples] = new_sample_weight # assign the new weight for new added samples
        self.weights.data[0] = new_gen_weight # assign the new weight for the generator
        self.samples[self.num_samples: self.num_samples + n_samples] = samples.cpu() # save samples
        self.num_samples += n_samples


    
    def get_samples_by_weights(self, n, generator):
        indices = list(WeightedRandomSampler(
            torch.nn.functional.softmax(self.weights[:self.num_samples], 0), n, replacement=True)) # weighted random samples

        indices = torch.tensor(indices)
        
        gen_indices = indices == 0 # filter out the index of generator (0)
        num_gen_indices = torch.sum(gen_indices) # get the number of samples needed from the generator
                
        selected_samples = self.samples[indices].to(self.device) # sample from self.samples
        
        if num_gen_indices > 0:
            # Sample from generator if num_gen_indices is larger than 0
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (num_gen_indices, opt.latent_dim))))
            # Generate a batch of images
            gen_imgs = generator(z)
            selected_samples[gen_indices] = gen_imgs.detach() # replace with samples generated by generator
        
        # return samples, corresponding weights and the weight for the generator (weighted by n/self.num_samples)
        return selected_samples
    
    def get_samples_uniformly(self, n, generator):
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        # Generate one image
        self.samples[0] = generator(z)[0].detach()

        # indices = torch.randperm(self.num_samples)[:n] # uniformly sample without replacement (TOO SLOW!!, replaced with the followings)
        # indices = self.unif[:self.num_samples].multinomial(n, replacement=False)
        # indices = list(WeightedRandomSampler(self.unif[:self.num_samples], n, replacement=False))
        indices = torch.randint(self.num_samples, (n,))
                
        softmax_weights = torch.nn.functional.softmax(self.weights[:self.num_samples], 0) # softmax of all
        selected_weights = softmax_weights[indices] # get the softmax values of selected indices
        sum_selected_weights = torch.sum(selected_weights)
        norm_selected_weights = selected_weights / sum_selected_weights # normalize weights to keep summation 1
        
        return self.samples[indices].to(self.device), norm_selected_weights, sum_selected_weights
        

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

        self.init_size = opt.img_size // 4
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

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss(reduction='none') # set reduction to none to let it return one loss value per sample, not per batch

# policy gradient loss
def pg_loss(pred, valid, weights):
    log_p = torch.log(weights)
    return -torch.mean(pred * log_p)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

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
saved_samples = SavedSamples(opt.n_epochs * dataloader.__len__() * opt.batch_size, "cuda" if cuda else "cpu")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Weights = torch.optim.Adam([saved_samples.weights], lr=opt.weight_lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

w_loss_list = []
w_grad_sum_list = []
g_loss_list = []
d_real_loss_list = []
d_fake_loss_list = []
w0_list = []

get_samples_by_weights_elapsed = 0
get_samples_uniformly_elapsed = 0

# ----------
#  Training
# ----------

for epoch in tqdm(range(opt.n_epochs)):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        
        # -----------------
        #  Train Weights
        # -----------------
        
        # disable the training of batchnorm 
        #generator.eval()
        #discriminator.eval()
        
        # clean the gradients of weights
        optimizer_Weights.zero_grad()
        weights_loss = None
        w_grad_sum = 0
        
        if saved_samples.num_samples > 1:
            # train the weights only when there are samples saved
            if opt.policy_loss:
                # sample by weights
                samples = saved_samples.get_samples_by_weights(imgs.shape[0], generator)
                weights = saved_samples.weights
                sum_weights = torch.sum(weights)
            else:
                # sample uniformly
                samples, weights, sum_weights = saved_samples.get_samples_uniformly(imgs.shape[0], generator)
            
            if not opt.skip_weights and not torch.isnan(weights)[0] and sum_weights > 1e-20: 
                # only train the weights when the sum of softmax weights (before normalization) is big enough to avoid gradient to become nan
                disc_pred = discriminator(samples)
                if opt.policy_loss:
                    # use pg_loss instead of adversarial_loss
                    weights_loss = pg_loss(disc_pred, valid, weights)
                else:
                    weights_loss = torch.sum(adversarial_loss(disc_pred, valid).squeeze() * weights) 
                
                weights_loss.backward()
                w_grad_sum = saved_samples.weights.grad.norm(dim=0, p=2).to('cpu') # save the norm of gradients for debugging
                optimizer_Weights.step()
                with torch.no_grad():
                    min_weight_derived_from_norm_min = torch.logsumexp(saved_samples.weights[1: saved_samples.num_samples], dim=0, keepdim=False) - torch.log(torch.tensor(1 / opt.min_gen_norm_weight - 1))
                    saved_samples.weights[0] = max([min_weight_derived_from_norm_min, saved_samples.weights[0], opt.min_gen_weight])
            
        # print("Train weights: ", saved_samples.weights[:saved_samples.num_samples])
            
        # -----------------
        #  Train Generator
        # -----------------
        
        #generator.train()
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(adversarial_loss(discriminator(gen_imgs), valid))

        g_loss.backward()
        optimizer_G.step()        

        # ---------------------
        #  Train Discriminator
        # ---------------------

        #generator.eval()
        #discriminator.train()
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss =  torch.mean(adversarial_loss(discriminator(real_imgs), valid))
        
        # Loss for fake images
        fake_imgs = saved_samples.get_samples_by_weights(imgs.shape[0], generator)
        fake_pred = discriminator(fake_imgs.detach())
        d_fake_loss = torch.mean(adversarial_loss(fake_pred, fake))
        
        d_loss = (real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        
        # -------------------------
        #  Save generated samples
        # -------------------------

		# Debug outputs
        if saved_samples.num_samples > 1:
            weights = torch.nn.functional.softmax(saved_samples.weights[:saved_samples.num_samples], 0)
            print(f'weights: {weights[1:].max()} | {weights[0].item()}')
        print(f'D for real: {discriminator(real_imgs).mean()}')
        print(f'D for fake: {discriminator(fake_imgs.detach()).mean()}')


        batches_done = epoch * len(dataloader) + i

        if epoch >= opt.sample_saving_delay and batches_done % opt.sample_saving_freq == 0:
            saved_samples.add_samples(gen_imgs.detach())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D real loss: %f] [D fake loss: %f] [G loss: %f] [W loss: %f] [W grad: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), real_loss.item(), d_fake_loss.item(), g_loss.item(), weights_loss.item() if weights_loss else 0, w_grad_sum)
        )
        
        w_loss_list.append(weights_loss.item() if weights_loss else 0)
        w_grad_sum_list.append(w_grad_sum)
        g_loss_list.append(g_loss.item())
        d_real_loss_list.append(real_loss.item())
        d_fake_loss_list.append(d_fake_loss.item())
        # w0_list.append(saved_samples.weights[0].item())
        w0_list.append(torch.nn.functional.softmax(saved_samples.weights[:saved_samples.num_samples], 0))

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], SAVED_FOLDER + "/%d.png" % batches_done, nrow=5, normalize=True)
            save_image(fake_imgs.data[:25], SAVED_FOLDER + "/%d_combined.png" % batches_done, nrow=5, normalize=True)

            
# save losses to file and draw a plot
with open(curr_time + '/loss_lists.pkl', 'wb') as f:
    pickle.dump([g_loss_list, d_real_loss_list, d_fake_loss_list, w_loss_list, w_grad_sum_list], f)
    
# save final weights to file
with open(curr_time + "/weights.pkl", 'wb') as f:
    pickle.dump(saved_samples.weights.tolist() , f)
    
x_len = len(g_loss_list)

g_loss_plt = plt.figure("Loss")
ax1 = plt.subplot(4, 1, 1)
plt.plot(range(x_len), g_loss_list, label = "Generator loss")
plt.tick_params('x', labelbottom=False)
plt.legend()
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(range(x_len), d_real_loss_list, label = "Discriminator real loss")
plt.tick_params('x', labelbottom=False)
plt.legend()
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(range(x_len), d_fake_loss_list, label = "Discriminator fake loss")
plt.tick_params('x', labelbottom=False)
plt.legend()
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(range(x_len), w_loss_list, label = "Weights loss")
plt.legend()
plt.xlabel('iter')
plt.savefig(curr_time + "/loss_plot.png", format="png")

w_plt = plt.figure("weights")
plt.plot(range(x_len), w0_list, label = "w0")
plt.legend()
plt.xlabel('iter')
plt.savefig(curr_time + "/weights_plot.png", format="png")