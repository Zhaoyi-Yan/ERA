import os
import torch
import torch.nn as nn
import logging
import time
import random
import numpy as np
import copy
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger, init_distributed_mode
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

from collections import defaultdict


torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    args, args_text = parse_args()
    args.exp_dir = os.path.join(args.save_prefix_root, 'fixed_FV', args.experiment)

    '''distributed'''
    init_distributed_mode(args)
#     init_dist(args)
    init_logger(args)

    if args.isAccuX:
        from lib.models.GAP_net_multi_parallel_accuX import GAP_net_accuX as GAP_net
    else:
        from lib.models.GAP_net_multi_parallel import GAP_net

    # save args
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = \
        build_dataloader(args)

    '''build model'''
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()
    
    loss_gap_cls = copy.deepcopy(loss_fn)
    val_loss_fn = loss_fn
    if not isinstance(args.eval_alphas, list):
        new_eval_alphas = [float(x) for x in args.eval_alphas.split(",")]
        args.eval_alphas = new_eval_alphas

    if not isinstance(args.scales, list):
        new_scales = [float(x) for x in args.scales.split(",")]
        args.scales = new_scales

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        in_c_tec = teacher_model.fc.weight.shape[1]
        
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        teacher_model.cuda()


    model = build_model(args, args.model, in_c_tec=in_c_tec)
    logger.info(model)

    in_c_stu = model.fc.weight.shape[1]
    model.cuda()
    model = DDP(model,
                device_ids=[args.gpu],
                find_unused_parameters=False)

    if args.kd != '':
        # build kd loss
        from lib.models.losses.kd_loss_imagenet import KDLoss
        loss_fn = KDLoss(model, teacher_model, loss_fn, args.kd, args.student_module,
                         args.teacher_module, args.ori_loss_weight, args.kd_loss_weight, args=args)
    
    if args.gap_type == 'enc':
        model_gap = GAP_net(n_blocks=args.gap_n_blocks, in_c=in_c_stu, in_c_tec=in_c_tec, is_multi_fc=args.is_multi_fc, act_type=args.act_type, n_module=args.gap_n_module)
    elif args.gap_type == 'unet':
        model_gap = GAP_unet(n_blocks=args.gap_n_blocks, in_c=in_c_stu)
    elif args.gap_type == 'vqnet':
        model_gap = GAP_net_vq(n_blocks=args.gap_n_blocks, in_c=in_c_stu, args=args)
    else:
        raise ValueError('-')
    logger.info(model_gap)


    if not args.up_dynamic:
        if in_c_stu != in_c_tec:
            model.module.up_fc.requires_grad = False
            model_gap.up_fc_list.requires_grad = False
        
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')
    
    logger.info(
        f'GAP Model created, filtered up_fc params: {model_gap.train_params() / 1e6:.3f} M, ')
    logger.info(
        f'GAP Model created, not filtered up_fc params: {model_gap.train_params(avoid=False) / 1e6:.3f} M, ')

    model_gap.cuda()
    model_gap = DDP(model_gap,
                    device_ids=[args.gpu],
                    find_unused_parameters=False)
    

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
        model_gap_ema = ModelEMA(model_gap, decay=args.model_ema_decay)
    else:
        model_ema = None
        model_gap_ema = None

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

    '''build scheduler'''
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(args.sched,
                                optimizer,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)
    scheduler_gap = build_scheduler(args.sched,
                                optimizer_gap,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)

    dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
        loss_scaler_gap = torch.cuda.amp.GradScaler()

    else:
        loss_scaler = None
        loss_scaler_gap = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     gap_model=model_gap,
                                     optimizer_gap=optimizer_gap,
                                     ema_model_gap=model_gap_ema,
                                     teacher_head=teacher_model.fc,
                                     save_dir=args.exp_dir,
                                     keep_num=args.keep_num,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'scaler_gap': loss_scaler_gap,
                                         'dyrep': dyrep
                                     },
                                    eval_alphas=args.eval_alphas,
                                    )

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
            scheduler_gap.finished = True

        scheduler.step(start_epoch * len(train_loader))
        scheduler_gap.step(start_epoch * len(train_loader))

        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )
    else:
        start_epoch = 0


    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs
                model_gap.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs

        # train
        teacher_head = teacher_model.fc
        metrics = train_epoch(args, epoch, model, model_ema, model_gap, model_gap_ema, train_loader,
                              optimizer, optimizer_gap, loss_fn, scheduler, scheduler_gap,
                              dyrep, loss_scaler, loss_scaler_gap, teacher_head, loss_gap_cls)       

        # validate
        if epoch >= args.start_eval_epoch:
            test_metrics = validate(args, epoch, model, val_loader, val_loss_fn, model_gap, teacher_head)
            if model_ema is not None:
                test_metrics = validate(args,
                                        epoch,
                                        model_ema.module,
                                        val_loader,
                                        val_loss_fn,
                                        model_gap=model_gap_ema.module if model_gap_ema else None,
                                        teacher_head=teacher_head if teacher_head else None,
                                        log_suffix='(EMA)')
            metrics.update(test_metrics)
            ckpts = ckpt_manager.update(epoch, metrics)
            logger.info('\n'.join(['Checkpoints:'] + [
                '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
            ]))


def train_epoch(args,
                epoch,
                model,
                model_ema,
                model_gap,
                model_gap_ema,
                loader,
                optimizer,
                optimizer_gap,
                loss_fn,
                scheduler,
                scheduler_gap=None,
                dyrep=None,
                loss_scaler=None,
                loss_scaler_gap=None,
                teacher_head=None,
                loss_gap_cls=None
                ):
    loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    model_gap.train()
    for batch_idx, (input, target) in enumerate(loader):
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=False) for speedup
        for p in model.parameters():
            p.grad = None
        
        for p in model_gap.parameters():
            p.grad = None

        if not args.kd:
            output = model(input)
            loss = loss_fn(output, target)
        else:
            """
            0. feat distill: `pre_fc_stu_up` ---> `pre_fc_tec`
            1. delta_x :  pre_fc_tec - pre_fc_stu_up
            2. pred_delta_x_list: for each output of the gap model, up_fc has been adapted inside gap model
            
            """
            loss, pre_fc_stu, pre_fc_stu_up, pre_fc_tec, (logits, logits_tec, kd_loss) = loss_fn(input, target, return_residual=True, return_kd_stuff=True)
            # this script for imagenet, feat distillation loss is computed here instead of in kd_loss.py
            feat_loss = torch.nn.Parameter(torch.tensor([0.0])).to(pre_fc_stu)
            delta_x = pre_fc_tec - pre_fc_stu_up
            
            if args.feat_distill_weight > 0:
                delta_loss_fn = nn.MSELoss()
                feat_loss = delta_loss_fn(pre_fc_stu_up, pre_fc_tec.detach()) * args.feat_distill_weight
            
            # cls loss for 0-th estimation
            loss_feat_cls = nn.Parameter(torch.tensor([0.0])).to(input)
            if args.featcls:
                mimic_tec_logit0 = teacher_head(pre_fc_stu_up)
                cur_loss_gap_cls = copy.deepcopy(loss_gap_cls)
                loss_feat_cls = cur_loss_gap_cls(mimic_tec_logit0, target) * args.ori_loss_weight
            
            # KL loss for 0-th estimation
            loss_feat_kd = nn.Parameter(torch.tensor([0.0])).to(input)
            if args.featKL:
                cur_kd_loss = copy.deepcopy(kd_loss)
                loss_feat_kd = cur_kd_loss(mimic_tec_logit0, logits_tec.detach()) * args.kd_loss_weight
            
            if args.featKL and args.featcls:
                loss = loss + feat_loss + (loss_feat_cls + loss_feat_kd)/2
            else:
                loss = loss + feat_loss + loss_feat_cls + loss_feat_kd
            
            # detach the gradients flow to the stu
            if args.gap_model_detach:
                pre_fc_stu_detach = pre_fc_stu.detach()
                pre_fc_stu_up_detach = pre_fc_stu_up.detach()
            else:
                pre_fc_stu_detach = pre_fc_stu
                pre_fc_stu_up_detach = pre_fc_stu_up
            
            pred_delta_x_list = model_gap(pre_fc_stu_detach)
            
            loss_delta_x = nn.Parameter(torch.tensor([0.0])).to(input)
            gap_cls_loss = nn.Parameter(torch.tensor([0.0])).to(input)
            if args.featKL:
                gap_kd_loss = nn.Parameter(torch.tensor([0.0])).to(input)
            
            target_delta_x = [] # delta_x_1, delta_x_1 - delta_x2, ...
            estimate_tec_feat_list = []  # x_s + delta_x_1, x_s + delta_x_1 + delta_x_2, ...
            estimate_tec_feat_list.append(pre_fc_stu_up_detach + pred_delta_x_list[0])
            # teacher head is not updating here, setting gradient=True is just make the gradient flow works
            # the optimizer/optimzier_gap does not contains the parameters of teacher_head
#             teacher_head.requires_grad=True
            
            for i in range(args.gap_n_module):
                delta_loss_fn = nn.MSELoss()
                cur_loss_gap_cls = copy.deepcopy(loss_gap_cls)
                cur_loss_gap_kd = copy.deepcopy(kd_loss)
                
                
                if i == 0: # the first block
                    loss_delta_x = loss_delta_x + delta_loss_fn(pred_delta_x_list[0], delta_x.detach()) * args.gap_feat_weight  * args.scales[i] # try add scales...
                    target_delta_x.append(delta_x)
                else:
                    target_delta_x.append(target_delta_x[i-1] - pred_delta_x_list[i-1])
                    loss_delta_x = loss_delta_x + delta_loss_fn(pred_delta_x_list[i], target_delta_x[i].detach()) * args.gap_feat_weight  * args.scales[i] # try add scales...

                # additional cls loss on estimated_tec_feat
                mimic_tec_logit = teacher_head(estimate_tec_feat_list[i])
                
                # just using the same weight of ori_loss
                gap_cls_loss = gap_cls_loss + cur_loss_gap_cls(mimic_tec_logit, target) * args.ori_loss_weight * args.scales[i]
                
                # just using the same weight of kd_loss
                if args.featKL:
                    gap_kd_loss = gap_kd_loss + cur_loss_gap_kd(mimic_tec_logit, logits_tec.detach()) * args.kd_loss_weight * args.scales[i]
                
                if i == args.gap_n_module - 1:
                    break
                else:
                    # prepare data for next estimation
                    estimate_tec_feat_list.append(estimate_tec_feat_list[i] + pred_delta_x_list[i+1])

            if args.featKL:
                loss_delta_x = loss_delta_x + (gap_cls_loss + gap_kd_loss) / 2
            else:
                loss_delta_x = loss_delta_x + gap_cls_loss
            loss_delta_x = loss_delta_x / args.gap_n_module

        if loss_scaler is None:
            (loss + loss_delta_x).backward()

        else:
            loss_scaler.scale(loss).backward(retain_graph=True)
            loss_scaler_gap.scale(loss_delta_x).backward()
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
                loss_scaler_gap.unscale_(optimizer_gap)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)
            torch.nn.utils.clip_grad_norm_(model_gap.parameters(),
                                           args.clip_grad_max_norm)

        if loss_scaler is None:
            optimizer.step()
            optimizer_gap.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()
            loss_scaler_gap.step(optimizer_gap)
            loss_scaler_gap.update()

        if model_ema is not None:
            model_ema.update(model)
            model_gap_ema.update(model_gap)

        loss_m.update(loss.item() + loss_delta_x.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s'.format(
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            lr=optimizer.param_groups[0]['lr'],
                            batch_time=batch_time_m,
                            data_time=data_time_m))
        scheduler.step(epoch * len(loader) + batch_idx + 1)
        scheduler_gap.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()

    return {'train_loss': loss_m.avg}


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
                    
                    # TODO: using feat distill fc
                    if args.final_out_type == 'avg':
                        mimic_tec_feat = pre_feat_up + torch.mean(torch.stack(pred_delta_x_list), dim=0)
                    elif args.final_out_type == 'acc':
                        mimic_tec_feat = pre_feat_up + torch.sum(torch.stack(pred_delta_x_list), dim=0)
                    else:
                        raise ValueError('ss')
                    
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
    
    if not teacher_head:
        return {'test_loss': loss_m.avg, 'top1': top1_m_dict[0].avg, 'top5': top5_m_dict[0].avg}
    
    result = {'test_loss': loss_m.avg} # Note: loss is not important, so just use one loss... 
    for k in args.eval_alphas:
        result[f'top1_{k}'] = top1_m_dict[k].avg
        result[f'top5_{k}'] = top5_m_dict[k].avg
    return result


if __name__ == '__main__':
    main()
