from torch.autograd import Variable
from torch import nn
import argparse
from preprocess import get_loader, init_weights
from torchvision.utils import save_image
from accelerate import Accelerator
from tqdm import tqdm
from model.U_Net import U_Net
from model.discriminator import Discriminator
from model.U2_Net import U2NET
from model.pix2pixHD import Generator_HD
from loss_function import *
import os
import config
import math
import pretty_errors

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()

params = config.parse_args()


def main():
    # Directories for loading data and saving results
    save_dir = 'results/' + params.dataset + '_results/'
    model_dir = 'results/' + params.dataset + '_model/'
    epoch_series = save_dir + 'epoch_series'
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(epoch_series):
        os.mkdir(epoch_series)

    # Data pre-processing
    train_dir = 'dataset/' + params.dataset + '/train/'
    valid_dir = 'dataset/' + params.dataset + '/valid/'
    train_data_loader = get_loader(image_path=train_dir,
                                   image_size=params.resolution,
                                   batch_size=params.batch_size,
                                   shuffle=True,
                                   num_workers=params.num_workers,
                                   mode='train',
                                   augmentation_prob=params.augmentation_prob)

    valid_data_loader = get_loader(image_path=valid_dir,
                                   image_size=params.resolution,
                                   batch_size=params.validation_size,
                                   shuffle=False,
                                   num_workers=params.num_workers,
                                   mode='valid',
                                   augmentation_prob=0.)
    valid_input, valid_target = valid_data_loader.__iter__().__next__()

    # Models
    if params.RGB is True:
        input_dim = 3
        output_dim = 3
    else:
        input_dim = 1
        output_dim = 1

    '''
    network種類選択及初期化
    '''
    if params.generator == 'U-Net':
        G = U_Net(input_dim=input_dim, num_filter=64, output_dim=output_dim)
    elif params.generator == 'U2-Net':
        G = U2NET(in_ch=input_dim, out_ch=output_dim)
    elif params.generator == 'pix2pixHD':
        G = Generator_HD(input_dim=input_dim, num_filter=64, output_dim=output_dim)
    D = Discriminator(input_dim * 2, params.ndf, 1)

    G.to(device)
    D.to(device)
    G.apply(init_weights)
    D.apply(init_weights)
    # Set the logger
    D_log_dir = save_dir + 'D_logs'
    G_log_dir = save_dir + 'G_logs'
    if not os.path.exists(D_log_dir):
        os.mkdir(D_log_dir)
    # D_logger = Logger(D_log_dir)

    if not os.path.exists(G_log_dir):
        os.mkdir(G_log_dir)
    # G_logger = Logger(G_log_dir)

    # Loss function
    '''      
    損失関数
    '''
    BCE_loss = torch.nn.BCELoss().to(device)
    if params.L1_type == 'Charbonnier_loss':
        L1_loss = L1_Charbonnier_loss().to(device)
    elif params.L1_type == 'L1_loss':
        L1_loss = nn.L1Loss().to(device)
    elif params.L1_type == 'L2_loss':
        L1_loss = nn.MSELoss().to(device)

    # Optimizers
    if params.WGANGP is False:
        learning_rate = params.lrG
    else:
        learning_rate = params.lrG
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(params.beta1, params.beta2))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(params.beta1, params.beta2))

    # Training GAN
    D_avg_losses = []
    G_avg_losses = []

    step = 0
    for epoch in range(params.num_epochs):
        D_losses = []
        G_losses = []

        # training
        for (input, target) in tqdm(train_data_loader):
            # input & target image data
            x_ = Variable(input.to(device))
            y_ = Variable(target.to(device))

            if params.WGANGP is False:
                '''
                普通のGAN
                '''
                # Train discriminator with real data
                D_real_decision = D(x_, y_).squeeze()
                real_ = Variable(torch.ones(D_real_decision.size()).to(device))
                D_real_loss = BCE_loss(D_real_decision, real_)

                # Train discriminator with fake data
                gen_image = G(x_)
                D_fake_decision = D(x_, gen_image).squeeze()
                fake_ = Variable(torch.zeros(D_fake_decision.size()).to(device))
                D_fake_loss = BCE_loss(D_fake_decision, fake_)

                # Back propagation
                D_loss = (D_real_loss + D_fake_loss) * 0.5
            else:
                '''
                WGAN-GP
                '''
                D_real_decision = D(x_, y_).squeeze()
                gen_image = G(x_)
                D_fake_decision = D(x_, gen_image).squeeze()
                D_loss_GP = torch.mean(D_fake_decision - D_real_decision)
                penalty = gradient_penalty(D, x_, y_, gen_image)
                D_loss = D_loss_GP + penalty

            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()
            # Train generator
            gen_image = G(x_)
            D_fake_decision = D(x_, gen_image).squeeze()
            if params.WGANGP is False:
                '''
                普通のGAN
                '''
                G_fake_loss = BCE_loss(D_fake_decision, real_)
            else:
                '''
                WGAN-GP
                '''
                G_fake_loss = torch.mean(-D_fake_decision)
            # L1 loss
            l1_loss = params.lamb * L1_loss(gen_image, y_)
            # Back propagation
            G_loss = G_fake_loss + l1_loss
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

            # ============ TensorBoard logging ============#
            # D_logger.scalar_summary('losses', D_loss.data[0], step + 1)
            # G_logger.scalar_summary('losses', G_loss.data[0], step + 1)
            step += 1

        D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

        # avg loss values for plot
        D_avg_losses.append(D_avg_loss)
        G_avg_losses.append(G_avg_loss)
        print('Epoch [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch + 1, params.num_epochs, D_avg_loss, G_avg_loss))

        # Show result for test image
        with torch.no_grad():
            valid_output = G(Variable(valid_input.to(device)))
        valid_target = valid_target.to(device)
        valid_output = valid_output.to(device)
        fake_images = torch.cat((valid_output[0:1, :, :, :], valid_target[0:1, :, :, :]), dim=0)
        for i in range(params.validation_size - 1):
            fake_images = torch.cat(
                (fake_images, valid_output[1 + i:2 + i, :, :, :], valid_target[1 + i:2 + i, :, :, :]), dim=0)
        # valid_target
        save_image(fake_images.data,
                   os.path.join(save_dir + 'epoch_series/', '{}_epoch.png'.format(epoch + 1)), nrow=8, scale_each=True)

    # Save trained parameters of model
    torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
    torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')


if __name__ == '__main__':
    main()
