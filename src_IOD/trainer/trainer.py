from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .losses import FocalLoss, RegL1Loss,STAloss,STAloss2

from progress.bar import Bar
from utils.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        [output] = self.model(batch['input'])
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats

class TrainLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TrainLoss, self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.crit_STAloss = STAloss(opt)
        self.crit_STAloss2 = STAloss2(opt)
        self.opt = opt

    def forward(self, output, batch):
        opt = self.opt
        output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)

        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        mov_loss = self.crit_mov(output['mov'], batch['mask'],
                                 batch['index'], batch['mov'])

        wh_loss = self.crit_wh(output['wh'], batch['mask'],
                               batch['index'], batch['wh'],
                               index_all=batch['index_all'])

        sta_sin_loss,sta_cos_loss = self.crit_STAloss(batch['centerKpoints'], batch['wh'],
                                output['hm'],output['wh'],output['STA_offset'],
                                batch['mask'],batch['index'],
                               index_all=batch['index_all'])

        # sta_sin_loss2,sta_cos_loss2 = self.crit_STAloss2(batch['centerKpoints'], batch['wh'],
        #                                               output['hm'],output['wh'],output['STA_offset'],
        #                                               batch['mask'],batch['index'],
        #                                               index_all=batch['index_all'])

        if self.opt.loss_option == 'STAloss':
            loss = opt.hm_weight * hm_loss.mean() + opt.wh_weight * wh_loss.mean() + opt.sta_weight * (sta_sin_loss.mean()+sta_cos_loss.mean())
        else:
            loss = opt.hm_weight * hm_loss.mean() + opt.wh_weight * wh_loss.mean() + opt.mov_weight * mov_loss.mean()

        loss = loss.unsqueeze(0)
        hm_loss = hm_loss.unsqueeze(0)
        wh_loss = wh_loss.unsqueeze(0)
        mov_loss = mov_loss.unsqueeze(0)
        sta_sin_loss = sta_sin_loss.mean().unsqueeze(0)
        sta_cos_loss = sta_cos_loss.mean().unsqueeze(0)
        # sta_sin_loss2 = sta_sin_loss2.mean().unsqueeze(0)
        # sta_cos_loss2 = sta_cos_loss2.mean().unsqueeze(0)
        # print(sta_cos_loss.detach().cpu().numpy(),"==",sta_cos_loss2.detach().cpu().numpy())
        # print(sta_sin_loss.detach().cpu().numpy(),"==",sta_sin_loss2.detach().cpu().numpy())

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,'wh_loss': wh_loss,'mov_loss':mov_loss,'sta_sin_loss':sta_sin_loss,'sta_cos_loss':sta_cos_loss}

        return loss, loss_stats


class Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'hm_loss',  'wh_loss','sta_sin_loss','sta_cos_loss'] if self.opt.loss_option == 'STAloss'\
                    else  ['loss', 'hm_loss',  'wh_loss','mov_loss']
        self.model_with_loss = ModleWithLoss(model, TrainLoss(opt))

    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    def run_epoch(self, phase, epoch, data_loader, writer):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar(opt.exp_id, max=num_iters)

        for iter, batch in enumerate(data_loader):
            if iter >= num_iters:
                break

            for k in batch:
                if k == 'input':
                    for i in range(len(batch[k])):
                        batch[k][i] = batch[k][i].to(device=opt.device, non_blocking=True)
                else:
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)

            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            step = iter // opt.visual_per_inter + num_iters // opt.visual_per_inter * (epoch - 1)

            for l in self.loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'][0].size(0))

                if phase == 'train' and iter % opt.visual_per_inter == 0 and iter != 0:
                    writer.add_scalar('train/{}'.format(l), avg_loss_stats[l].avg, step)
                    writer.flush()
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            bar.next()
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # MODIFY for pytorch 0.4.0
                    state[k] = v.to(device=device, non_blocking=True)
                    # state[k] = v.to(device=device)
