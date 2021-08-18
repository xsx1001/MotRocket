import time
import torch
from progress.bar import Bar
from data_parallel import DataParallel
from utils import AverageMeter
import torch.nn as nn
from losses import FocalLoss
from losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def gen_oracle_map(feat, ind, w, h):
    # feat: B x maxN x featDim
    # ind: B x maxN
    batch_size = feat.shape[0]
    max_objs = feat.shape[1]
    feat_dim = feat.shape[2]
    out = np.zeros((batch_size, feat_dim, h, w), dtype=np.float32)
    vis = np.zeros((batch_size, h, w), dtype=np.uint8)
    ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(batch_size):
        queue_ind = np.zeros((h*w*2, 2), dtype=np.int32)
        queue_feat = np.zeros((h*w*2, feat_dim), dtype=np.float32)
        head, tail = 0, 0
        for j in range(max_objs):
            if ind[i][j] > 0:
                x, y = ind[i][j] % w, ind[i][j] // w
                out[i, :, y, x] = feat[i][j]
                vis[i, y, x] = 1
                queue_ind[tail] = x, y
                queue_feat[tail] = feat[i][j]
                tail += 1
        while tail - head > 0:
            x, y = queue_ind[head]
            f = queue_feat[head]
            head += 1
            for (dx, dy) in ds:
                xx, yy = x + dx, y + dy
                if xx >= 0 and yy >= 0 and xx < w and yy < h and vis[i, yy, xx] < 1:
                    out[i, :, yy, xx] = f
                    vis[i, yy, xx] = 1
                    queue_ind[tail] = xx, yy
                    queue_feat[tail] = f
                    tail += 1
    return out

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class BaseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(BaseLoss, self).__init__()
        self.crit = FocalLoss()
        # self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
        #       RegLoss() if opt.reg_loss == 'sl1' else None
        # self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
        #       NormRegL1Loss() if opt.norm_wh else \
        #       RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            # if not opt.mse_loss:
            output = _sigmoid(output)

            # if opt.eval_oracle_hm:
            #     output['hm'] = batch['hm']
        #   if opt.eval_oracle_wh:
        #       output['wh'] = torch.from_numpy(gen_oracle_map(
        #       batch['wh'].detach().cpu().numpy(), 
        #       batch['ind'].detach().cpu().numpy(), 
        #       output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
        #   if opt.eval_oracle_offset:
        #       output['reg'] = torch.from_numpy(gen_oracle_map(
        #       batch['reg'].detach().cpu().numpy(), 
        #       batch['ind'].detach().cpu().numpy(), 
        #       output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output, batch['hm']) / opt.num_stacks
        #   if opt.wh_weight > 0:
        #       if opt.dense_wh:
        #           mask_weight = batch['dense_wh_mask'].sum() + 1e-4
        #           wh_loss += (
        #               self.crit_wh(output['wh'] * batch['dense_wh_mask'],
        #               batch['dense_wh'] * batch['dense_wh_mask']) / 
        #               mask_weight) / opt.num_stacks
        #       elif opt.cat_spec_wh:
        #           wh_loss += self.crit_wh(
        #               output['wh'], batch['cat_spec_mask'],
        #               batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        #       else:
        #           wh_loss += self.crit_reg(
        #               output['wh'], batch['reg_mask'],
        #               batch['ind'], batch['wh']) / opt.num_stacks
          
        #   if opt.reg_offset and opt.off_weight > 0:
        #       off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
        #                         batch['ind'], batch['reg']) / opt.num_stacks
            
    #   loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
    #         opt.off_weight * off_loss
        loss = hm_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss}
        return loss, loss_stats

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss
    
    def forward(self, batch):
        outputs = self.model(batch["input"])
        # img = outputs[0].cpu().detach().numpy()
        # print(img.shape)
        # # img = np.transpose(img, (1,2,0))
        # # cv2.imshow("", img)
        # plt.imshow(img[0], cmap="gray")
        # plt.show()
        # cv2.waitKey(0)
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class BaseTrainer(object):
    def __init__(
      self, opt, model, optimizer=None):
      self.opt = opt
      self.optimizer = optimizer
      self.loss_stats, self.loss = self._get_losses(opt)
      self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, gpus, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)
      
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results_num = 0
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}/{}'.format("rocket", opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            batch_new = {}
            batch_new["input"] = batch[0]
            batch_new["hm"] = batch[1]
            batch = batch_new
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
            else:
                bar.next()
            
            # if opt.debug > 0:
            #     self.debug(batch, output, iter_id)
          
            if opt.test:
                results_num = self.save_result(opt, output, batch, results_num)
            del output, loss, loss_stats
    
        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results
  
    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, opt, output, batch, results_num):
        output_path = opt.save_dir
        # print(output.shape)
        for i in range(batch["input"].shape[0]):
            result = output[i].cpu().detach().numpy()
            gt = batch["hm"][i].cpu().detach().numpy()[-1, :, :]
            ori = np.mean(batch["input"][i].cpu().detach().numpy(), axis=0)
            # ori_reshape = np.transpose(batch["input"][i].cpu().detach().numpy(), (1,2,0))
            # print(result.dtype)
            # plt.imshow(result, cmap="gray")
            # plt.show()
            # plt.imshow(ori_reshape, cmap="gray")
            # plt.show()
            # plt.imshow(gt, cmap="gray")
            # plt.show()
            # print(result.shape)
            # print(gt.shape)
            # print(ori.shape)
            # print(result.dtype)
            # print(max(map(max, result)))
            # print(max(map(max, gt)))
            # info = np.iinfo(result.dtype) # Get the information of the incoming image type
            # print(result.shape)
            result = result / max(map(max, result))# normalize the data to 0 - 1
            result = 255 * result # Now scale by 255
            result = result.astype(np.uint8)
            ret, binary_result = cv2.threshold(result, max(map(max, result))-10, 255, cv2.THRESH_BINARY)
            # result = np.uint8(result)
            # info = np.iinfo(gt.dtype) # Get the information of the incoming image type
            gt = gt / max(map(max, gt)) # normalize the data to 0 - 1
            gt = 255 * gt # Now scale by 255
            gt = gt.astype(np.uint8)

            # info = np.iinfo(ori.dtype) # Get the information of the incoming image type
            ori = ori / max(map(max, ori)) # normalize the data to 0 - 1
            ori = 255 * ori # Now scale by 255
            ori = ori.astype(np.uint8)
            # ret, ori = cv2.threshold(ori, max(map(max, ori))-10, 255, cv2.THRESH_BINARY)
            # info = np.iinfo(ori_reshape.dtype) # Get the information of the incoming image type
            # ori_reshape = ori_reshape / max(map(max, ori_reshape.any())) # normalize the data to 0 - 1
            # ori_reshape = 255 * ori_reshape # Now scale by 255
            # ori_reshape = ori_reshape.astype(np.uint8)

            # gt = np.uint8(gt)
            # ori = np.uint8(ori)
            # ori_reshape = np.uint8(ori_reshape)
            result00 = np.concatenate((result, binary_result), axis=1)
            result11 = np.concatenate((gt, ori), axis=1)
            result1 = np.concatenate((result00, result11), axis=0)
            # plt.imshow(ori_reshape)
            # # plt.show()
            cv2.imwrite(os.path.join(output_path, "result", str(results_num)+".jpg"), result1)
            # cv2.imwrite(os.path.join(output_path, "result_output", str(results_num)+".jpg"), result)
            # cv2.imwrite(os.path.join(output_path, "result_gt", str(results_num)+".jpg"), batch["hm"][i].cpu().detach().numpy()[0])
            # cv2.imwrite(os.path.join(output_path, "result_ori", str(results_num)+".jpg"), ori_reshape)
            results_num += 1

        return results_num


    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss']
        loss = BaseLoss(opt)
        return loss_states, loss
  
    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)