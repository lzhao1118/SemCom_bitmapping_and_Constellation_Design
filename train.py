import torch
import torch.optim as optim
import time
import numpy as np
import random
import traceback
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from utils import AverageMeter, save_model


def train_one_epoch(net, train_loader, optimizer, calcu_ssim, writer, config,
                    epoch, cur_lr, global_step, args, batch_snr):
    """Train for one epoch."""
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]

    total_loss = 0
    batch_count = 0
    results = np.zeros((len(train_loader), 4))
    for batch_idx, data in enumerate(train_loader):
        start_time = time.time()
        global_step += 1
        input = data.cuda() if isinstance(data, torch.Tensor) else data[0].cuda()
        recon_image, CBR, SNR, mse, loss_G = net(input, batch_snr)
        loss = loss_G
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新指标
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        cbrs.update(CBR)
        snrs.update(SNR)

        if mse.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
            psnrs.update(psnr.item())
            msssim = 1 - calcu_ssim(input, recon_image.clamp(0., 1.)).mean().item()
            msssims.update(msssim)
        else:
            psnrs.update(100)
            msssims.update(100)

        total_loss += loss.item()
        batch_count += 1

        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            print(' | '.join([
                f'Epoch {epoch}',
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Time {elapsed.val:.3f}',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                f'Lr {cur_lr}',
            ]))
        results[batch_idx - 1] = [mse.item(), CBR, psnr.item(), msssim]

    cbr_, psnr_, ssim_ = results[:, 1].mean(), results[:, 2].mean(), results[:, 3].mean()
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')

    writer.add_scalar('Train/PSNR', psnr_, epoch)
    writer.add_scalar('Train/SSIM', ssim_, epoch)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

    print(f"Epoch completed with average loss: {avg_loss}")

    return avg_loss, global_step


def test(net, test_loader, calcu_ssim, writer, epoch, cur_lr, config, args, batch_snr):
    """Periodic test during training."""
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, batch_snr)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - calcu_ssim(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    print(' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, batch_snr)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - calcu_ssim(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    print(' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    writer.add_scalar('Test/PSNR', results_psnr[0], epoch)
    writer.add_scalar('Test/SSIM', results_msssim[0], epoch)

    print("SNR: {}".format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}".format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


def run_training(args, config, net, train_loader, test_loader, calcu_ssim):
    """Run the full training loop with optimizer, scheduler, writer setup."""
    model_params = [{'params': net.parameters(), 'lr': args.lr}]
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    print(f"Steps per epoch: {steps_epoch}")

    # 设置学习率调度器
    print("Setting up learning rate scheduler...")
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.decay)

    # 设置早停
    best_loss = float('inf')
    no_improve = 0
    min_lr = args.min_lr

    if args.pretrain == '':
        if args.pass_channel == False:
            writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/' +
                '%s_C%s_lr%s_snr%s_%s/' % (args.px, args.C, args.lr, args.snr_min, args.snr_max))
        if args.pass_channel == True:
            writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/' +
                '%s_C%s_lr%s_pc_snr%s_%s/' % (args.px, args.C, args.lr, args.snr_min, args.snr_max))
    elif args.pretrain != '':
        if args.pass_channel == False:
            writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/' +
                '%s_Pre_C%s_lr%s_snr%s_%s/' % (args.px, args.C, args.lr, args.snr_min, args.snr_max))
        if args.pass_channel == True:
            writer = SummaryWriter(log_dir='./runs_tb_kodak/BitSC_Prob/' +
                '%s_Pre_C%s_lr%s_pc_snr%s_%s/' % (args.px, args.C, args.lr, args.snr_min, args.snr_max))

    print("=== Starting training loop ===")
    for epoch in range(steps_epoch, config.tot_epoch):
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        batch_snr = random.randint(args.snr_min, args.snr_max)
        try:
            # 训练一个epoch
            train_loss, global_step = train_one_epoch(
                net, train_loader, optimizer, calcu_ssim, writer, config,
                epoch, cur_lr, global_step, args, batch_snr)
            # 学习率调度
            scheduler.step()
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']
                if param_group['lr'] < min_lr:
                    param_group['lr'] = min_lr

            # 定期保存和测试
            if (epoch + 1) % config.save_model_freq == 0:
                print(f"=== Running periodic test at epoch {epoch + 1} ===")
                save_model(net, save_path=config.models + '/{}_EP{}.model'
                            .format(config.filename, epoch + 1))
                test(net, test_loader, calcu_ssim, writer, epoch, cur_lr, config, args, batch_snr)

        except Exception as e:
            print(f"Error during epoch {epoch}: {str(e)}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise e
