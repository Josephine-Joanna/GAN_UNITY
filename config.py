import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='Cell2SF', help='入力種類:'
                                                                             'Cell2SF'
                                                                             'cityscape'
                                                                             'Wrinkle_force_microscopy'
                                                                             'wrinkle_extraction'
                                                                             'Stress_fiber_extraction'
                                                                             'Contour_extraction')
    parser.add_argument('--batch_size', type=int, default=4, help='訓練batch_size')
    parser.add_argument('--validation_size', type=int, default=64, help='訓練中評価数')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=256, help='解析度')
    parser.add_argument('--num_epochs', type=int, default=300, help='訓練epoch数')
    parser.add_argument('--num_workers', type=int, default=0, help='CPU核心数')
    parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--lamb', type=float, default=100, help='L1損失係数')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--augmentation_prob', type=float, default=1.0, help='augmentation')
    parser.add_argument('--RGB', type=bool, default=False,  help='是否RGB')
    parser.add_argument('--WGANGP', type=bool, default=True, help='是否WGAN-GP')
    parser.add_argument('--multi_D', type=bool, default=False, help='是否多尺度判別器')
    parser.add_argument('--generator', type=str, default='pix2pixHD', help='ネットワーク種類:'
                                                                           'U-Net'
                                                                           'U2-Net'
                                                                           'pix2pixHD'
                                                                           'Transformer'
                                                                           'AttU_Net'
                                                                           'SW-UNet')
    parser.add_argument('--L1_type', type=str, default='L1_loss', help='L1損失関数種類:'
                                                                       'Charbonnier_loss'
                                                                       'L1_loss'
                                                                       'L2_loss'
                                                                       'Binary_cross_entropy')

    opt = parser.parse_args()

    return opt
