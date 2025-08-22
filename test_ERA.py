import os
import torch
import torch.nn as nn
import logging
import time
import numpy as np
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
from lib.utils.optim import build_optimizer

from lib.models.builder import build_model
from lib.models.MBRNet import GAP_net
from lib.dataset.builder import build_dataloader
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger, init_distributed_mode
from lib.utils.misc import accuracy, AverageMeter, CheckpointManager
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = args.experiment

    args.resume = os.path.join(args.exp_dir, args.resume_ckpt)

    '''distributed'''
    init_distributed_mode(args)
    init_logger(args, eval=True)

    '''build dataloader'''
    _, val_dataset, _, val_loader = \
        build_dataloader(args, only_eval=True)

    '''build model'''
    loss_fn = nn.CrossEntropyLoss().cuda()
    val_loss_fn = loss_fn

    if not isinstance(args.eval_alphas, list):
        new_eval_alphas = [float(x) for x in args.eval_alphas.split(",")]
        args.eval_alphas = new_eval_alphas

    # get the teacher head
    teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
    logger.info(
        f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
    teacher_model.cuda()
    teacher_head = copy.deepcopy(teacher_model.fc)
    in_c_tec = teacher_model.fc.weight.shape[1]
    del teacher_model
        
    model =  build_model(args, args.model, in_c_tec=in_c_tec)
    logger.info(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    in_c_stu = model.fc.weight.shape[1]

    model_gap = GAP_net(n_blocks=args.gap_n_blocks, in_c=in_c_stu, in_c_tec=in_c_tec, is_multi_fc=args.is_multi_fc, act_type=args.act_type, n_module=args.gap_n_module)
    logger.info(model_gap)
    logger.info(
        f'GAP Model created, params: {get_params(model_gap) / 1e6:.3f} M, ')

    model.cuda()
    model_gap.cuda()
    model = DDP(model,
                device_ids=[args.gpu],
                find_unused_parameters=False)
    model_gap = DDP(model_gap,
                device_ids=[args.gpu],
                find_unused_parameters=False)


    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
        ema_model_gap = ModelEMA(model_gap, decay=args.model_ema_decay)
    else:
        model_ema = None
        ema_model_gap = None

    '''build optimizer'''
    optimizer = build_optimizer(args.opt,
                                model.module,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)
    optimizer_gap = build_optimizer(args.opt,
                                model_gap.module,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)


    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     gap_model=model_gap,
                                     optimizer_gap=optimizer_gap,
                                     ema_model_gap=ema_model_gap,
                                     teacher_head=teacher_head,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     eval_alphas=args.eval_alphas,
                                    )

    if args.resume:
        epoch = ckpt_manager.load(args.resume)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'epoch {epoch}'
        )
    else:
        epoch = 0

    
    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn, model_gap, teacher_head)

    logger.info(test_metrics)

    

def validate(args, epoch, model, loader, loss_fn, model_gap=None, teacher_head=None, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m_dict = defaultdict(lambda: AverageMeter(dist=True))
    top5_m_dict = defaultdict(lambda: AverageMeter(dist=True))
    
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.eval()
    if teacher_head:
        model_gap.eval()
        teacher_head.eval()
    
    if teacher_head is None:
        args.eval_alphas = [0.0]
    
    for eval_alpha in args.eval_alphas:
        for batch_idx, (input, target) in enumerate(loader):
            with torch.no_grad():
                pre_feats, pre_feats_up, output = model(input, is_feat=True, out_up_feat=True)
                if teacher_head:
                    pre_feat = pre_feats[-1]
                    pre_feat_up = pre_feats_up[-1]
                    
                    pred_delta_x_list = model_gap(pre_feat)
                    mimic_tec_feat = pre_feat_up + torch.sum(torch.stack(pred_delta_x_list), dim=0)
                    
                    mimic_tec_logits = teacher_head(mimic_tec_feat)
                    output = output * (1 - eval_alpha) + eval_alpha * mimic_tec_logits
                loss = loss_fn(output, target)

            top1, top5 = accuracy(output, target, topk=(1, 5))
            loss_m.update(loss.item(), n=input.size(0))
            top1_m_dict[eval_alpha].update(top1 * 100, n=input.size(0))
            top5_m_dict[eval_alpha].update(top5 * 100, n=input.size(0))

            batch_time = time.time() - start_time
            batch_time_m.update(batch_time)
            if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
                logger.info('Test{}: {} [{:>4d}/{}] '
                            'Eval_alpha:{} '
                            'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                            'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                            'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                            'Time: {batch_time.val:.2f}s'.format(
                                log_suffix,
                                epoch,
                                batch_idx,
                                len(loader),
                                eval_alpha,
                                loss=loss_m,
                                top1=top1_m_dict[eval_alpha],
                                top5=top5_m_dict[eval_alpha],
                                batch_time=batch_time_m))
        start_time = time.time()
    # Create a result dictionary
    
    if not teacher_head:
        return {'test_loss': loss_m.avg, 'top1': top1_m_dict[0].avg, 'top5': top5_m_dict[0].avg}
    
    result = {'test_loss': loss_m.avg} # Note: loss is not important, so just use one loss... 
    for k in args.eval_alphas:
        result[f'top1_{k}'] = top1_m_dict[k].avg
        result[f'top5_{k}'] = top5_m_dict[k].avg
    return result


if __name__ == '__main__':
    main()
