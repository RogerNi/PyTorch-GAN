import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
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

#===========Decision Tree==========
import sys
from tree import tree
sys.modules['tree'] = tree
#==================================

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
SAVED_FOLDER = curr_time + "-images-sampled-on-prob"
os.makedirs(SAVED_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") # replace by --input_def
parser.add_argument("--input_def", type=str, default="[32,32,16,16,7]", help="input definitions (in bits)") # output is by default 1-D
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--sample_saving_delay", type=int, default=0, help="number of epochs to delay before starting to save samples")
parser.add_argument("--new_weight_ratio", type=float, default=0.5, help="The ratio of the new weights for samples to the weight for generator. Default: 0.5, new sample weights each have the same value as the new weight for the generator. If set to 0, the new sample weights are 0 while the generator weight remains unchanged.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="A small value to avoid weight from becoming 0")
parser.add_argument("--skip_weights", default=False, action="store_true", help="Whether to skip optimizing weights")
parser.add_argument("--disable_gpu", default=False, action="store_true", help="Whether to disable GPU")


opt = parser.parse_args()
print(opt)

input_def = [int(bits) for bits in opt.input_def[1: -1].split(',')]
total_bits = sum(input_def)

cuda = True if torch.cuda.is_available() and not opt.disable_gpu else False


class SavedSamples():
    def __init__(self, total_n, device):
        self.device = device
        self.weights = torch.ones(total_n + 1, requires_grad=True, device=torch.device(device)) # the first (0th) weight is for the generator
        self.unif = torch.ones(total_n + 1)
        self.samples = torch.zeros((total_n + 1, 2 * total_bits))
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
        generator.eval()
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
        generator.train()
        return selected_samples
    
    def get_samples_uniformly(self, n, generator):
        generator.eval()
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
        
        generator.train()
        
        return self.samples[indices].to(self.device), norm_selected_weights, sum_selected_weights
        


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
            nn.Linear(1024, 2 * total_bits),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # img = img.view(img.size(0), 2, total_bits)
        # return torch.argmax(img, dim=1).type(torch.get_default_dtype())
        img = img.view(img.size(0), total_bits, 2)
        img = nn.Softmax(dim=2)(img)
        img = img.view(img.size(0), -1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2 * total_bits, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)
        validity = self.model(img)

        return validity


# Loss function
# adversarial_loss = torch.nn.BCELoss(reduction='none') # set reduction to none to let it return one loss value per sample, not per batch

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()
    
# define dataset

class NetworkPacketDataset(Dataset):
    def __init__(self, size):
        self.data = (torch.rand((size, total_bits)) > 0.5).type(torch.get_default_dtype())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.__unpack__(index)
    
    def __unpack__(self, index):
        # unpack to 2 * total_bits (0 -> [1, 0], 1 -> [0, 1])
        return torch.stack([1- self.data[index], self.data[index]]).T.reshape(-1)

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    NetworkPacketDataset(int(1e4)),
    batch_size=opt.batch_size,
    shuffle=True,
)

# SavedSamples
saved_samples = SavedSamples(opt.n_epochs * dataloader.__len__() * opt.batch_size, "cuda" if cuda else "cpu")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Weights = torch.optim.Adam([saved_samples.weights], lr=100 * opt.lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

w_loss_list = []
w_grad_sum_list = []
g_loss_list = []
d_loss_list = []
# d_real_loss_list = []
# d_fake_loss_list = []

get_samples_by_weights_elapsed = 0
get_samples_uniformly_elapsed = 0

# ----------
#  Training
# ----------

for epoch in tqdm(range(opt.n_epochs)):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        
        
        # -----------------
        #  Train Weights
        # -----------------
        
        # disable the training of batchnorm 
        
        # clean the gradients of weights
        optimizer_Weights.zero_grad()
        weights_loss = None
        w_grad_sum = 0
        
        if saved_samples.num_samples > 1:
            # train the weights only when there are samples saved
            start = time.time()
            samples, weights, sum_weights = saved_samples.get_samples_uniformly(imgs.shape[0], generator)
            get_samples_uniformly_elapsed = time.time() - start
            
            if not opt.skip_weights and not torch.isnan(weights)[0] and sum_weights > 1e-20: 
                # only train the weights when the sum of softmax weights (before normalization) is big enough to avoid gradient to become nan
                disc_pred = discriminator(samples)
                weights_loss = -torch.sum(disc_pred * weights) 
                weights_loss.backward()
                w_grad_sum = saved_samples.weights.grad.norm(dim=0, p=2).to('cpu') # save the norm of gradients for debugging
                optimizer_Weights.step()
            
        # print("Train weights: ", saved_samples.weights[:saved_samples.num_samples])
            
        # -----------------
        #  Train Generator
        # -----------------
        
        # generator.train()
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = -torch.mean(discriminator(gen_imgs))

        g_loss.backward()
        optimizer_G.step()        

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # generator.eval()
        # discriminator.train()
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # real_loss =  torch.mean(adversarial_loss(discriminator(real_imgs), valid))
        
        # Loss for fake images
        start = time.time()
        fake_imgs = saved_samples.get_samples_by_weights(imgs.shape[0], generator)
        get_samples_by_weights_elapsed = time.time() - start
        fake_pred = discriminator(fake_imgs.detach())
        # d_fake_loss = torch.mean(adversarial_loss(fake_pred, fake))
        
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        d_loss.backward()
        optimizer_D.step()
        
        # -------------------------
        #  Save generated samples
        # -------------------------
        
        if epoch >= opt.sample_saving_delay:
            saved_samples.add_samples(gen_imgs.detach())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [W loss: %f] [W grad: %f] [elapse 1: %f] [elapse 2: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), weights_loss.item() if weights_loss else 0, w_grad_sum, get_samples_by_weights_elapsed, get_samples_uniformly_elapsed)
        )
        
        w_loss_list.append(weights_loss.item() if weights_loss else 0)
        w_grad_sum_list.append(w_grad_sum)
        g_loss_list.append(g_loss.item())
        d_loss_list.append(d_loss.item())
        # d_real_loss_list.append(real_loss.item())
        # d_fake_loss_list.append(d_fake_loss.item())
        
        
torch.save(saved_samples.samples, curr_time + '-samples.pt')

            
# save losses to file and draw a plot
with open(curr_time + '-loss_lists.pkl', 'wb') as f:
    pickle.dump([g_loss_list, d_loss_list, w_loss_list, w_grad_sum_list], f)
    
# save final weights to file
with open(curr_time + "-weights.pkl", 'wb') as f:
    pickle.dump(saved_samples.weights.tolist() , f)
    
x_len = len(g_loss_list)
plt.plot(range(x_len), g_loss_list, label = "Generator loss")
plt.plot(range(x_len), d_loss_list, label = "Discriminator loss")
# plt.plot(range(x_len), d_fake_loss_list, label = "Discriminator fake loss")
# plt.plot(range(x_len), w_loss_list, label = "Weights loss")
# plt.plot(range(x_len), w_grad_sum_list, label = "Sum of weights gradient")

plt.legend()
plt.xlabel('iter')
plt.ylabel('loss')
plt.savefig(curr_time + "-loss_plot.svg", format="svg")