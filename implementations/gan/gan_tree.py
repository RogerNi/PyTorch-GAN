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
import json

import random
import string

#===========Decision Tree==========
import sys
from tree import tree
sys.modules['tree'] = tree
dtree = pickle.loads(open("tree/acl5_10k-23-acc-27-bytes-1592431760.25.pkl", "rb").read())
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

curr_time = time.strftime("%Y%m%d-%H%M%S") + "-tree-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
os.makedirs(curr_time, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--weight_lr", type=float, default=0.02, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") # replace by --input_def
parser.add_argument("--input_def", type=str, default="[32,32,16,16,7]", help="input definitions (in bits)") # output is by default 1-D
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--sample_saving_delay", type=int, default=0, help="number of epochs to delay before starting to save samples")
parser.add_argument("--sample_saving_freq", type=int, default=500, help="frequency (iterations) to save samples")
parser.add_argument("--new_weight_ratio", type=float, default=0.5, help="The ratio of the new weights for samples to the weight for generator. Default: 0.5, new sample weights each have the same value as the new weight for the generator. If set to 0, the new sample weights are 0 while the generator weight remains unchanged.")
parser.add_argument("--epsilon", type=float, default=1e-10, help="A small value to avoid weight from becoming 0")
parser.add_argument("--skip_weights", default=False, action="store_true", help="Whether to skip optimizing weights")
parser.add_argument("--disable_gpu", default=False, action="store_true", help="Whether to disable GPU")
parser.add_argument("--min_gen_weight", type=float, default=torch.finfo().min, help="the lower bound of raw sampling weight of the generator")
parser.add_argument("--min_gen_norm_weight", type=float, default=0, help="the lower bound of normalized sampling weight of the generator, valid range: [0, 1)")
parser.add_argument("--policy_loss", default=False, help="whether to use policy gradient loss instead of binary cross entropy loss")

parser.add_argument("--tree_threshold", type=float, default=4, help="threshold for decision tree")
parser.add_argument("--load_dataset", type=str, default="", help="path of dataset to load")
parser.add_argument("--init_data_size", type=int, default=1e5, help="size of data to generate initially")
parser.add_argument("--fixed_w0", type=float, default=0, help="whether to use fixed w0, valid range: [0, 1] with 0 being w0 not fixed")


opt = parser.parse_args()
print(opt)
with open(curr_time + '/commandline_args.txt', 'w') as f:
    json.dump(opt.__dict__, f, indent=2)

input_def = [int(bits) for bits in opt.input_def[1: -1].split(',')]
total_bits = sum(input_def)

cuda = True if torch.cuda.is_available() and not opt.disable_gpu else False
print("cuda: ", cuda)


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
        if self.num_samples == 1:
            all_weights = torch.tensor([1], device=self.device)
        elif opt.fixed_w0 == 0:
            all_weights = torch.nn.functional.softmax(self.weights[:self.num_samples], 0)
        else:
            all_weights = torch.cat((torch.tensor([opt.fixed_w0], device=self.device), torch.nn.functional.softmax(self.weights[1:self.num_samples], 0) * (1 - opt.fixed_w0)), 0)
            
        # assert abs(torch.sum(all_weights) - 1) < 1e-3, "Sum of weights is not close to 1 which is {}".format(torch.sum(all_weights))
        
        indices = list(WeightedRandomSampler(all_weights, n, replacement=True)) # weighted random samples

        indices = torch.tensor(indices)
        
        gen_indices = indices == 0 # filter out the index of generator (0)
        num_gen_indices = torch.sum(gen_indices) # get the number of samples needed from the generator
                
        selected_samples = self.samples[indices].to(self.device) # sample from self.samples
        weights = all_weights[indices]
        
        if num_gen_indices > 0:
            # Sample from generator if num_gen_indices is larger than 0
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (num_gen_indices, opt.latent_dim))))
            # Generate a batch of images
            generator.eval()
            gen_imgs = generator(z)
            generator.train()
            selected_samples[gen_indices] = gen_imgs.detach() # replace with samples generated by generator
        
        # return samples, corresponding weights and the weight for the generator (weighted by n/self.num_samples)
        return selected_samples, weights
    
    def get_samples_uniformly(self, n, generator):
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
        # Generate one image
        generator.eval()
        self.samples[0] = generator(z)[0].detach()
        generator.train()

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

# policy gradient loss
def pg_loss(pred, weights):
    log_p = torch.log(weights)
    return -torch.mean(pred.flatten() * log_p)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    # adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

    
print(generator)
print("Generator device:" + str(next(generator.parameters()).is_cuda))
print(discriminator)
print("Discriminator device:" + str(next(discriminator.parameters()).is_cuda))

#================Define Dataset================

class BaseNetworkPacketDataset(Dataset):
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.__unpack__(index)
    
    def __unpack__(self, index):
        # unpack to 2 * total_bits (0 -> [1, 0], 1 -> [0, 1])
        return torch.stack([1- self.data[index], self.data[index]]).T.reshape(-1)

class NetworkPacketDataset(BaseNetworkPacketDataset):
    def __init__(self, size):
        self.data = (torch.rand((size, total_bits)) > 0.5).type(torch.get_default_dtype())
    
class NetworkPacketDatasetUnbalancedSample(BaseNetworkPacketDataset):
    def __init__(self, size):
        self.data = torch.zeros((size, 0))
        input_prob = [0.1, 0.3, 0.5, 0.7, 0.9]
        assert len(input_prob) == len(input_def)
        for id, prob in zip(input_def, input_prob):    
            self.data = torch.hstack([self.data, (torch.rand((size, id)) > prob).type(torch.get_default_dtype())]) 
    
class NetworkPacketRareDataset(BaseNetworkPacketDataset):
    def __init__(self, size, threshold):
        curr_len = 0
        pbar = tqdm(total=size, desc="Generating rare packets")
        while curr_len < size:
            packet = []
            for id in input_def:    
                packet.append((torch.rand(id) > 0.5).type(torch.get_default_dtype()))
            converted_packet = tuple([self._b2i(p.numpy().astype("int")) for p in packet])
            if dtree.match(converted_packet) > threshold:
                self.data = torch.vstack([self.data, torch.hstack(packet)]) if curr_len > 0 else torch.hstack(packet)
                curr_len += 1
                pbar.update(1)
        pbar.close()
                
    def _b2i(self, b_list):
        return int("".join(str(x) for x in b_list), 2) 
    
class NetworkPacketRareDatasetFromFile(BaseNetworkPacketDataset):
    def __init__(self, path):
        self.data = torch.load(path)
    
#==============================================
    
# dataset = NetworkPacketDataset(int(1e5))
if opt.load_dataset:
    dataset = NetworkPacketRareDatasetFromFile(opt.load_dataset)
else:
    dataset = NetworkPacketRareDataset(opt.init_data_size, 4)
    torch.save(dataset.data, curr_time + "/init_dataset.pt")

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# SavedSamples
saved_samples = SavedSamples(- opt.n_epochs * dataloader.__len__() * (- opt.batch_size // opt.sample_saving_freq), "cuda" if cuda else "cpu")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_Weights = torch.optim.Adam([saved_samples.weights], lr=opt.weight_lr, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

w_loss_list = []
w_grad_sum_list = []
g_loss_list = []
d_loss_list = []
w0_list = []

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
            if opt.policy_loss:
                # sample by weights
                samples, weights = saved_samples.get_samples_by_weights(imgs.shape[0], generator)
                # Debug outputs
                print(f'Sampled weights: {weights.max()}')
                sum_weights = torch.sum(weights)
            else:
                # sample uniformly
                samples, weights, sum_weights = saved_samples.get_samples_uniformly(imgs.shape[0], generator)
            
            if not opt.skip_weights and not torch.isnan(weights)[0] and sum_weights > 1e-20: 
                # only train the weights when the sum of softmax weights (before normalization) is big enough to avoid gradient to become nan
                disc_pred = discriminator(samples)
                if opt.policy_loss:
                    # use pg_loss instead of adversarial_loss
                    weights_loss = pg_loss(disc_pred, weights)
                else:
                    weights_loss = -torch.sum(disc_pred * weights) 
                
                weights_loss.backward()
                w_grad_sum = saved_samples.weights.grad.norm(dim=0, p=2).to('cpu') # save the norm of gradients for debugging
                optimizer_Weights.step()
                with torch.no_grad():
                    min_weight_derived_from_norm_min = torch.logsumexp(saved_samples.weights[1: saved_samples.num_samples], dim=0) - torch.log(torch.tensor(1 / opt.min_gen_norm_weight - 1)) if opt.min_gen_norm_weight > 0 else torch.finfo().min
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
        g_loss = -torch.mean(discriminator(gen_imgs))

        g_loss.backward()
        optimizer_G.step()        

        # ---------------------
        #  Train Discriminator
        # ---------------------

        #generator.eval()
        #discriminator.train()
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # real_loss =  torch.mean(adversarial_loss(discriminator(real_imgs), valid))
        
        # Loss for fake images
        fake_imgs, _ = saved_samples.get_samples_by_weights(imgs.shape[0], generator)
        fake_pred = discriminator(fake_imgs.detach())

        # d_fake_loss = torch.mean(adversarial_loss(fake_pred, fake))
        d_real_loss = -torch.mean(discriminator(real_imgs))
        d_fake_loss = torch.mean(discriminator(fake_imgs))
        
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        optimizer_D.step()
        
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
        
        # -------------------------
        #  Save generated samples
        # -------------------------
        batches_done = epoch * len(dataloader) + i

        if epoch >= opt.sample_saving_delay and batches_done % opt.sample_saving_freq == 0:
            saved_samples.add_samples(gen_imgs.detach())

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D real loss: %f] [D fake loss: %f] [G loss: %f] [W loss: %f] [W grad: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_real_loss.item(), d_fake_loss.item(), g_loss.item(), weights_loss.item() if weights_loss else 0, w_grad_sum)
        )
        
        w_loss_list.append(weights_loss.item() if weights_loss else 0)
        w_grad_sum_list.append(w_grad_sum)
        g_loss_list.append(g_loss.item())
        d_loss_list.append(d_loss.item())
        # d_real_loss_list.append(real_loss.item())
        # d_fake_loss_list.append(d_fake_loss.item())
        w0_list.append(torch.nn.functional.softmax(saved_samples.weights[:saved_samples.num_samples], 0)[0].cpu().detach())

        
        
torch.save(saved_samples.samples[1:saved_samples.num_samples], curr_time + '/samples.pt')

            
# save losses to file and draw a plot
with open(curr_time + '/loss_lists.pkl', 'wb') as f:
    pickle.dump([g_loss_list, d_loss_list, w_loss_list, w_grad_sum_list], f)
    
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
plt.plot(range(x_len), d_loss_list, label = "Discriminator loss")
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
