import torch.optim
from F_fct_module import *
from torch.optim.lr_scheduler import LambdaLR


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0

    warm_up_steps 预热阶段的训练步数，在此期间学习率从初始值线性增长至最大值。
    lr_min 余弦衰减阶段的最低学习率，衰减过程不会低于该值。
    lr_max 余弦衰减阶段的最高学习率，预热结束后学习率从此值开始衰减。
    lr_start 预热阶段的初始学习率，通常低于 lr_max 以稳定训练初期参数。
    max_decay_steps 总训练步数（包含预热阶段），用于计算余弦衰减周期长度。
    verbosity_interval 日志输出间隔步数，若设为 0 则不输出训练信息。
    """
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n,**kwargs)



class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
：
    warm_up_steps (列表) 每个周期对应的预热步数列表，长度与周期数一致。
    f_min (列表) 各周期余弦衰减阶段的最低学习率列表。
    f_max (列表) 各周期余弦衰减阶段的最高学习率列表。
    f_start (列表) 各周期预热阶段的初始学习率列表。
    cycle_lengths (列表) 每个周期的总步数列表，用于确定各阶段时间跨度。
    verbosity_interval 控制日志输出频率。
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                    1 + np.cos(t * np.pi))
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)



class LambdaLinearScheduler(LambdaWarmUpCosineScheduler):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f




class LambdaLinearScheduler2(LambdaWarmUpCosineScheduler2):

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: print(f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                                                       f"current cycle {cycle}")

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (self.cycle_lengths[cycle] - n) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f



def optimizer_scheduler(parameter=None,
                        lr=1e-3,
                        weight_decay=1e-2,
                        batas=(0.9, 0.999),
                        eps=1e-8,
                        opt=None,
                        sched=None,
                        sched_=None,
                        warm_step=5000,
                        warm_step2=[5000],
                        start=[1e-5],
                        start2=[1e-5],
                        min=1e-8,
                        min2=[1e-8],
                        max=1e-3,
                        max2=[1e-3],
                        lr_step=10000,
                        lr_step2=[10000],
                        v_i=0):
    assert  opt in ('adade_lta', 'adagrad', 'adam', 'adamw', 'sparse_adam', 'adamax', 'asgd', 'sgd', 'radam', 'rprop', 'rmsprop', 'optimizer', 'nadam', 'lbfgs')
    assert  sched in ('LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ConstantLR', 'LinearLR', 'ExponentialLR', 'SequentialLR', 'CosineAnnealingLR', 'ChainedScheduler', 'ReduceLROnPlateau', 'CyclicLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'PolynomialLR')
    assert  sched_ in ('Cos', 'Cos_Crcl', 'Lnr', 'Lnr_Crcl')

    if opt is not None:
        if opt=='AdamW':
            optimizer = torch.optim.AdamW(parameter, lr=lr, weight_decay=weight_decay, betas=batas, eps=eps)
        elif opt == 'Adam':
            optimizer = torch.optim.Adam(parameter, lr=lr, weight_decay=weight_decay, betas=batas, eps=eps)
            # optimizer = torch.optim.Adam(parameter,lr=lr_d, betas=(0.5, 0.9))
        elif opt == 'SGD':
            optimizer = torch.optim.SGD(parameter, lr=lr, weight_decay=weight_decay)
        else:
            assert 0==1,f'undefined optimizer with opt: {opt}'


    if sched is not None:
        if sched == 'LambdaLR':
            if sched_ == 'Cos':
                scheduler_ = LambdaWarmUpCosineScheduler(warm_up_steps=warm_step, lr_min=min, lr_max=max, lr_start=start, max_decay_steps=lr_step, verbosity_interval=v_i)
                scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule)
            elif sched_ == 'Cos_Crcl':
                scheduler_ = LambdaWarmUpCosineScheduler2(warm_up_steps=warm_step2, f_min=min2, f_max=max2, f_start=start2, cycle_lengths=lr_step2, verbosity_interval=v_i)
                scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule)
            elif sched_ == 'Lnr':
                scheduler_ = LambdaLinearScheduler(warm_up_steps=warm_step, lr_min=min, lr_max=max, lr_start=start, max_decay_steps=lr_step, verbosity_interval=v_i)
                scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule)
            elif sched_ == 'Lnr_Crcl':
                scheduler_ = LambdaLinearScheduler2(warm_up_steps=warm_step2, f_min=min2, f_max=max2, f_start=start2, cycle_lengths=lr_step2, verbosity_interval=v_i)
                scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule)
        # if sched == 'LambdaLR':
        else:
            assert 0 == 1, f'undefined scheduler with sched: {sched}'


    return optimizer,scheduler


if __name__ == "__main__":

    opt,schd = optimizer_scheduler()