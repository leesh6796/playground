import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # setting gpu number

# load packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio #### install with "pip install imageio"
from IPython.display import HTML

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import make_grid

from fid_score import calculate_fid_given_paths  # The code is downloaded from github


version = "ver1"
dir_img = f"./img_{version}"
dir_real = f"{dir_img}/real"
dir_fake = f"{dir_img}/fake"

image_size = 64
batch_size = 128
workers = 4
ngpu = 4


def save_gif(training_progress_images, images):
    '''
        training_progress_images: list of training images generated each iteration
        images: image that is generated in this iteration
    '''
    img_grid = make_grid(images.data)
    img_grid = np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0))
    img_grid = 255. * img_grid 
    img_grid = img_grid.astype(np.uint8)
    training_progress_images.append(img_grid)
    imageio.mimsave(f"{dir_img}/training_progress.gif", training_progress_images)
    return training_progress_images


# visualize gif file
def vis_gif(training_progress_images):
    fig = plt.figure()
    
    ims = []
    for i in range(len(training_progress_images)):
        im = plt.imshow(training_progress_images[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    
    html = ani.to_html5_video()
    HTML(html)


# visualize gif file
def plot_gif(training_progress_images, plot_length=10):
    plt.close()
    fig = plt.figure()
    
    total_len = len(training_progress_images)
    for i in range(plot_length):
        im = plt.imshow(training_progress_images[int(total_len/plot_length)*i])
        plt.show()


def save_image_list(dataset, real):
    if real:
        base_path = f"./img_{version}/real"
    else:
        base_path = f"./img_{version}/fake"
    
    dataset_path = []
    
    for i in range(len(dataset)):
        save_path =  f'{base_path}/image_{i}.png'
        dataset_path.append(save_path)
        vutils.save_image(dataset[i], save_path)
    
    return base_path


nc = 3 # number of channels, RGB
nz = 100 # input noise dimension
ngf = 64 # number of generator filters
ndf = 64 #number of discriminator filters

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


if __name__ == "__main__":
    # Create folders
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')
        
    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
        
    if not os.path.exists(dir_img):
        os.mkdir(dir_img)
        
    if not os.path.exists(dir_real):
        os.mkdir(dir_real)

    if not os.path.exists(dir_fake):
        os.mkdir(dir_fake)

    dataset = dset.ImageFolder(root="data_faces",
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    lr = 0.0002
    Train = False

    netG = Generator(ngpu).cuda()
    netD = Discriminator(ngpu).cuda()
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

    if not Train:
        netG.load_state_dict(torch.load(f"./checkpoint/netG_{version}_epoch_50.pth"))
        netD.load_state_dict(torch.load(f"./checkpoint/netD_{version}_epoch_50.pth"))
        print(netG.eval())
        print(netD.eval())

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    if Train:
        print("Start Training")
        fixed_noise = torch.randn(batch_size, 100, 1, 1).cuda()
        criterion = nn.BCELoss()
        n_epoch = 100
        training_progress_images_list = []
        for epoch in range(n_epoch):
            if (epoch + 1) >= 50 and (epoch+1) % 20 == 0:
                optimizerG.param_groups[0]['lr'] /= 2
                optimizerD.param_groups[0]['lr'] /= 2
                print("learning rate change!")

            for i, (data, _) in enumerate(dataloader):
                ####################################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
                ###################################################
                # train with real
                netD.zero_grad()
                data = data.cuda()
                batch_size = data.size(0)
                label = torch.ones((batch_size,)).cuda()
                output = netD(data)
                errD_real = criterion(output, label)
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, 100, 1, 1).cuda()
                fake = netG(noise)
                label = torch.zeros((batch_size,)).cuda()
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                D_G_z1 = output.mean().item()
                
                # Loss backward
                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()

                ########################################
                # (2) Update G network: maximize log(D(G(z))) #
                ########################################
                netG.zero_grad()
                label = torch.ones((batch_size,)).cuda()  # fake labels are real for generator cost
                output = netD(fake)
                errG = criterion(output, label)
                D_G_z2 = output.mean().item()

                errG.backward()
                optimizerG.step()
                
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' 
                    % (epoch+1, n_epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            #save the output
            fake = netG(fixed_noise)
            training_progress_images_list = save_gif(training_progress_images_list, fake)  # Save fake image while training!
            
            # Check pointing for every epoch
            if (epoch + 1) % 5 == 0:
                torch.save(netG.state_dict(), f"./checkpoint/netG_{version}_epoch_{epoch+1}.pth")
                torch.save(netD.state_dict(), f"./checkpoint/netD_{version}_epoch_{epoch+1}.pth")
                print(f"Save checkpoint at epoch {epoch+1}")
                # f"./checkpoint/netG_{version}_epoch_{epoch+1}.pth"

    # TEST
    print("Start Test")
    TB = 1000
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=TB, shuffle=True, num_workers=workers)

    for i, (data, _) in enumerate(dataloader):
        real_dataset = data
        break

    noise = torch.randn(TB, nz, 1, 1).cuda()
    fake_dataset = netG(noise)

    print("Generate real/fake images")
    real_image_path_list = save_image_list(real_dataset, True)
    fake_image_path_list = save_image_list(fake_dataset, False)

    print("Evaluate FID score")
    fid_value = calculate_fid_given_paths([real_image_path_list, fake_image_path_list], 1000, False, 2048)

    print (f'FID score: {fid_value}')