from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#DeepOBS imports
from deepobs.pytorch.datasets.afhq import afhq  # Import the data loading module of DeepOBS
from deepobs import pytorch as pt
from deepobs import config
from deepobs.pytorch.testproblems import afhq_dcgan
from deepobs.pytorch.testproblems import testproblem, testproblems_utils, testproblems_modules

DATA_DIR = "../data_deepobs"

# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of training epochs
num_epochs = 10




data = afhq(batch_size=128)
# Create an instance of the FMNIST Data class (which is a subclass of the Data Set class of DeepOBS), using for example 64 as the batch size.

next_batch = next(iter(data._train_dataloader))  # get the next batch of the training data set. If you replace '_train_dataloader', with '_test_dataloader' you would get a batch of the test data set and so on.

print(len(next_batch))
print(len(next_batch[0]))


# Plot some trainig images from the next_batch
# To check wether default device can be used
# Will be deleted later
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(next_batch[0].to(testproblem.config.get_default_device())[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
#plt.show()

# Variables needed for training
device = testproblem.config.get_default_device()


# Set up test problem
testproblem = afhq_dcgan(batch_size=128)
testproblem.set_up()
# Use training data set
testproblem.train_init_op()

"""
Training Loop

 Split up into two parts:
    1. Update D 
    2. Update G

 1. Discriminator
 Maximize log(D(x))+log(1-D(G(z)))
 Using seperate mini-batches for real and fake samples
     1. real sample batch with forward pass through D
       calculate loss (log(D(x)), calculate gradients with  backward pass
     2. Same for fake batch sample, with loss (log(1-D(G(z))))
     3. Accumulate gradients with backward pass
     4. Update D's parameters with an optimizer step


 2. Generator
 Modified function for G: Maximize (log(D(G(z)))
     1. Classify G's output from part 1 with D
     2. Compute G's loss using real labels as GT
       compute G's gradients with backward pass
     3. Update G's parameters with an optimizer step


 Report training statistics
 Push fixed_noise batch through G to visually track the progress
     Loss_D sum of losses for all real and all fake batches
     Loss_G modified loss function of G
     D(x) average output of D for the all real batch. Should start close to 1 and 
        in theory converge to 0.5 as G gets better
     D(G(z)) average of D's output for fake batch. First number represents D before update, second after D's update
        Should start near 0 then converge to 0.5 as G gets better
"""
# Input vector for G
fixed_noise = torch.randn(64, testproblem.generator.nz, 1, 1, device=device)
# Print random vector
#print(fixed_noise)
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(testproblem.net.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(testproblem.generator.parameters(), lr=lr, betas=(beta1, 0.999))


# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
D_acc_real = []
D_acc_fake = []

iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs+1):
    # For each batch in the dataloader
    for i, data in enumerate(testproblem.data._train_dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        # Train with all-real batch
        testproblem.net.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = testproblem.net(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = testproblem.loss_function(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, testproblem.generator.nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = testproblem.generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = testproblem.net(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = testproblem.loss_function(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        testproblem.generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = testproblem.net(fake).view(-1)
        # Calculate G's loss based on this output
        errG = testproblem.loss_function(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats

        if i % 50 == 0:
            print('Evaluating after %d of %d epochs and %d of %d iterations...\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(testproblem.data._train_dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_acc_real.append(D_x)
        D_acc_fake.append(D_G_z2)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs - 1) and (i == len(testproblem.data._train_dataloader) - 1)):
            with torch.no_grad():
                fake = testproblem.generator(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            plt.title("Fake image G(z)")
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True)))
            plt.savefig('results/images/afhq_dcgan_fakes_['+str(epoch)+']['+str(iters)+']')

        iters += 1

# plot loss
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.suptitle("G and D Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
# plot accuracy
plt.subplot(1, 2, 2)
plt.suptitle("Discriminator Accuracy for Real and Fake Images")
plt.plot(D_acc_real, label="real")
plt.plot(D_acc_fake, label="fake")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('results/plots/afhq_dcgan[epochs: '+str(num_epochs)+'][batch_size: '+str(len(next_batch[0]))+']')
