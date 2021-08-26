import os

import torch
import torch.utils.data
from opts import opts
from model import create_model, load_model, save_model
# from models.data_parallel import DataParallel
from logger import Logger
from base_trainer import BaseTrainer
from get_dataset import get_dataset

# from datasets.dataset_factory import get_dataset
# from trains.train_factory import train_factory


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # opt.device = torch.device('cpu')
    
    print('Creating model...')
    model = create_model(opt.arch)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model_path = "./exp/default/model_best.pth"
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']

        model.load_state_dict(state_dict, strict=False)

    trainer = BaseTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    dataset_val, dataset_train, dataset_test = get_dataset(opt)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=1, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    if opt.test:
        _, preds = trainer.val(0, test_loader)
        # test_loader.dataset.run_eval(preds, opt.save_dir)
        return

    

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                    epoch, model, optimizer)
        with torch.no_grad():
            log_dict_val, preds = trainer.val(epoch, val_loader)
        for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if log_dict_val[opt.metric] < best:
            best = log_dict_val[opt.metric]
            save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                    epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                    epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                    epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)