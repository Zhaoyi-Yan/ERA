import os
import time
import shutil
import logging
import subprocess
import torch


def init_dist(args):
    args.distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    if args.slurm:
        args.distributed = True
    if not args.distributed:
        # task with single GPU also needs to use distributed module
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        args.local_rank = 0
        args.distributed = True

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        if args.slurm:
            # processes are created with slurm
            proc_id = int(os.environ['SLURM_PROCID'])
            ntasks = int(os.environ['SLURM_NTASKS'])
            node_list = os.environ['SLURM_NODELIST']
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f'scontrol show hostname {node_list} | head -n1')
            os.environ['MASTER_ADDR'] = addr
            os.environ['WORLD_SIZE'] = str(ntasks)
            args.local_rank = proc_id % num_gpus
            os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
            os.environ['RANK'] = str(proc_id)
            print(f'Using slurm with master node: {addr}, rank: {proc_id}, world size: {ntasks}')

        os.environ['MASTER_PORT'] = args.dist_port
        args.device = 'cuda:%d' % args.local_rank
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if not args.slurm:
            torch.cuda.set_device(args.rank)
        print(f'Training in distributed model with multiple processes, 1 GPU per process. Process {args.rank}, total {args.world_size}.')
    else:
        print('Training with a single process on 1 GPU.')

def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

# create logger file handler for rank 0,
# ignore the outputs of the other ranks
def init_logger(args, eval=False):
    logger = logging.getLogger()
    if args.rank == 0:
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        logger.setLevel(logging.INFO)
        if not eval:
            log_name = f'log_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.txt'
        else:
            log_name = 'eval.txt'
        fh = logging.FileHandler(os.path.join(args.exp_dir, log_name))
        fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
        logger.info(f'Experiment directory: {args.exp_dir}')

    else:
        logger.setLevel(logging.ERROR)
