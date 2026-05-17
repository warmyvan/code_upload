import torch
import torch.nn as nn
import numpy as np
from inspect import isfunction
import torch.nn.functional as F
from operator import itemgetter
import os
import math
from einops import repeat
import json
import random
import psutil
from tqdm import tqdm
import yaml
import pandas as pd
# from skimage.morphology import (ball)
from quantimpy import morphology as mp
from quantimpy import minkowski as mk

# # # # # def fct # # # # #

def cmpt_mk(input_tensor,mean=0.5):
    
    # 确保输入形状为 (96, 96, 96)
    if input_tensor.shape == (1, 1, 96,96,96):
        tensor = input_tensor.reshape(96, 96, 96)
    elif input_tensor.shape == (96, 96, 96):
        tensor = input_tensor.copy()
    else:
        raise ValueError("输入张量形状必须为 (1, 1, 96, 96, 96) 或 (96, 96, 96)")

#     binary = (tensor.cpu().numpy() < 0).astype(np.float32)
    binary = (tensor.cpu().numpy() < mean)
    if len(binary.shape) >3:
        binary = np.squeeze(binary)
    if binary.shape != (96, 96, 96):
        raise ValueError("转换后 binary 的形状必须为 (96, 96, 96)，当前形状为 {}".format(binary.shape))
    
    minkowski = mk.functionals(binary)
    print(f'mkvsk:{minkowski}')
    
    mk_v = minkowski[0]/(96*96*96)
    
    return minkowski[0],minkowski[1],minkowski[2],minkowski[3]




def cmpt_p(input_tensor,mean=0.5):
    """
    计算 3D 介质的孔隙度（整体及沿 x、y、z 轴切片）

    参数:
    input_tensor (numpy.ndarray): 输入的单通道 3D 张量，形状为 (1, 1, 96*96*96) 或 (96, 96, 96)

    返回:
    tuple:
        - 整体孔隙度张量（形状 (1, 1, 96*96*96)，所有值为整体孔隙度）
        - 沿 x 轴切片孔隙度数组（形状 (96,)）
        - 沿 y 轴切片孔隙度数组（形状 (96,)）
        - 沿 z 轴切片孔隙度数组（形状 (96,)）
    """
    # 确保输入形状为 (96, 96, 96)
    if input_tensor.shape == (1, 1, 96,96,96):
        tensor = input_tensor.reshape(96, 96, 96)
    elif input_tensor.shape == (96, 96, 96):
        tensor = input_tensor.copy()
    else:
        raise ValueError("输入张量形状必须为 (1, 1, 96, 96, 96) 或 (96, 96, 96)")

    binary = (tensor.cpu().numpy() < mean).astype(np.float32)
    
    print('binary:',binary.min(),binary.max())

    # 计算整体孔隙度
    overall_porosity = np.mean(binary)

    # 计算沿各轴的切片孔隙度
    x_slice_porosity = np.mean(binary, axis=(1, 2))  # 沿 x 轴切片（y-z 平面）
    y_slice_porosity = np.mean(binary, axis=(0, 2))  # 沿 y 轴切片（x-z 平面）
    z_slice_porosity = np.mean(binary, axis=(0, 1))  # 沿 z 轴切片（x-y 平面）

    print(overall_porosity, len(x_slice_porosity), len(y_slice_porosity), len(z_slice_porosity))
    print('pore:',overall_porosity, np.mean(x_slice_porosity), np.mean(y_slice_porosity), np.mean(z_slice_porosity)  )
    return overall_porosity, x_slice_porosity, y_slice_porosity, z_slice_porosity



def moment_sample(moment, seed=None, mode=False):
    posterior = DiagonalGaussianDistribution(moment)

    if mode==True:
        return posterior.mode()
    else:
        if seed is not None:
            set_seed(seed)
        return posterior.sample()


def moment_mode(moment):
    posterior = DiagonalGaussianDistribution(moment)
    return posterior.mode()


# # # # # def fct # # # # #
def exists(val):
    return val is not None



def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def cpu_gpu_allocate_print():
    # CPU GPU allocated #
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024)} MB")
    if torch.cuda.is_available():
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / (1024 * 1024)} MB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved() / (1024 * 1024)} MB")

        
def to_cpu_numpy(x):
    """严格类型检查的转换逻辑：
    1. torch.Tensor → CPU ndarray
    2. list → numpy array（任意元素类型）
    3. tuple → numpy array（必须全为数值类型）
    4. float/int → 转换为 numpy 标量（float64 / int64）
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, tuple):
        return np.array(x)
    elif isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (np.number, np.bool_)):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")    

        
def find_files_with_keywords(modelpath, key1, key2):
    """
    在指定路径下查找文件名包含 key1 和 key2 的文件，并返回完整路径。

    参数：
        modelpath (str): 文件夹路径。
        key1 (str): 第一个关键字。
        key2 (str): 第二个关键字。

    返回：
        list: 包含符合条件的文件路径列表。
    """
    # 检查路径是否存在
    if not os.path.exists(modelpath):
        raise FileNotFoundError(f"路径 {modelpath} 不存在！")

    # 获取路径下的所有文件
    all_files = os.listdir(modelpath)

    # 筛选文件名包含 key1 和 key2 的文件
    matched_files = [
        os.path.join(modelpath, fname)  # 返回完整路径
        for fname in all_files
        if key1 in fname and key2 in fname
    ]

    return matched_files
 
    
def find_files_with_keys_n(folder_path, key1, key2, num=None):
    """
    找到文件夹中文件名包含 key1 和 key2 的所有文件路径，并限制返回的数量。

    参数:
        folder_path (str): 文件夹路径。
        key1 (str): 第一个关键字。
        key2 (str): 第二个关键字。
        num (int, optional): 最大返回的文件路径数量。默认为 None（不限制）。

    返回:
        list: 包含符合条件的文件路径的列表。
    """
    matching_files = []

    # 递归遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # 检查文件名是否同时包含 key1 和 key2
            if key1 in file_name and key2 in file_name:
                # 构造完整文件路径并添加到结果列表
                full_path = os.path.join(root, file_name)
                matching_files.append(full_path)

                # 如果达到最大数量，提前返回
                if num is not None and len(matching_files) >= num:
                    return matching_files

    return matching_files
    


def generate_single_mask():
    
    """生成单个样本的三维区域掩码（无通道维度）"""
    x, y, z = np.indices((96, 96, 96))
    mask = np.zeros((96, 96, 96), dtype=np.int8)
    method = np.random.choice([1, 2, 3, 4])

    if method == 1: 
        mask[x < 32] = 1
        mask[(x >= 32) & (x < 64)] = 2
        mask[x >= 64] = 3
    elif method == 2: 
        mask[y < 32] = 1
        mask[(y >= 32) & (y < 64)] = 2
        mask[y >= 64] = 3
    elif method == 3: 
        mask[z < 32] = 1
        mask[(z >= 32) & (z < 64)] = 2
        mask[z >= 64] = 3
    elif method == 4: 
        region1 = (x < 32) | (x >= 64) | (y < 32) | (y >= 64) | (z < 32) | (z >= 64)
        remaining = ~region1
        region2 = remaining & (x < 48)
        region3 = remaining & (x >= 48)
        mask[region1] = 1
        mask[region2] = 2
        mask[region3] = 3
    return mask


def identify_method(mask):
    """
    判断输入的 mask 是由 generate_single_mask 中的哪种 method 生成的
    返回 1, 2, 3, 4 或 -1（表示不匹配）
    """

    mask = np.squeeze(mask)

    if mask.shape != (96, 96, 96):
        return -2

    # 获取所有非零坐标
    coords = np.nonzero(mask)
    labels = mask[coords]
    
  
    axes = ['x', 'y', 'z']
    ranges = [(0, 31), (32, 63), (64, 95)] 
    

    for axis in range(3):  # 0: x, 1: y, 2: z
        valid = True

        for label in [1, 2, 3]:

            axis_coords = coords[axis][labels == label]
            if len(axis_coords) == 0:
                valid = False
                break
            min_coord, max_coord = np.min(axis_coords), np.max(axis_coords)
            low, high = ranges[label-1]
            if min_coord < low or max_coord > high:
                valid = False
                break
        if valid:
            return axis + 1 

    x, y, z = np.indices((96, 96, 96))
    region1 = (x < 32) | (x >= 64) | (y < 32) | (y >= 64) | (z < 32) | (z >= 64)
    remaining = ~region1
    region2 = remaining & (x < 48)
    region3 = remaining & (x >= 48)
    
    if (np.array_equal(mask == 1, region1) and 
        np.array_equal(mask == 2, region2) and 
        np.array_equal(mask == 3, region3)):
        return 4

    return -1 


def generate_batch(properties, masks=None, is_train=True):

    batch_size = properties.shape[0]

    if masks is None:
        masks = np.stack([generate_single_mask() for _ in range(batch_size)], axis=0)

    keys = np.full((batch_size, 3), [1, 2, 3], dtype=np.int32) 
    values = np.zeros((batch_size, 3), dtype=np.float32)

    for i in range(batch_size):
        sample_properties = properties[i, 0]  # (96,96,96)
        sample_mask = masks[i]  # (96,96,96)

        if is_train:
            for idx, region in enumerate([1, 2, 3]):
                region_mask = (sample_mask == region)
                total = region_mask.sum()
                if total == 0:
                    values[i, idx] = 0.0
                else:
                    values[i, idx] = (sample_properties[region_mask] < 127.0).sum() / total
        else:
            print('eval:')
            for idx, region in enumerate([1, 2, 3]):
                region_mask = (sample_mask == region)
                total = region_mask.sum()
                if total == 0:
                    values[i, idx] = 0.0
                else:
                    
                    pore_count = (sample_properties[region_mask] < 0).sum()
                    values[i, idx] = pore_count / total

    return masks, keys, values   
    
    
    
    
    
    
def sample_z(sample, seed=None, eps=True):
    if seed is not None:
        np.random.seed(seed)
    
    mu = sample[:6, ...]  # (6,12,12,12)
    sigma = sample[6:, ...]
    epsilon = np.random.normal(size=mu.shape)
    if eps:
        z = mu + sigma * epsilon
    else:
        z = mu
    return z       
    
        

def generate_random_data_pair(dims,batch=1,rslut=96,chnl=1):

    if dims==3:
        x = torch.rand(batch, chnl, rslut, rslut, rslut)
    else:
        x = torch.rand(batch, chnl, rslut, rslut)

    x = (x * 255).to(torch.uint8)


    y = torch.randint(0, 2, (batch,))  

    return x, y
        

def convert_module_to_f16(x):
    pass



def convert_module_to_f32(x):
    pass



def count_flops_attn(model, _x, y):

    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])



def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module



def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params



# seed
def set_seed(seed: int):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 



# ema
class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)



# # # # # checkpoint # # # # #
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag: 
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)



class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        if torch.is_grad_enabled():
            ctx.training_state = {
                name: getattr(module, "training", False)
                for name, module in run_function.named_modules()
                if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout))
            }

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):

        for i, x in enumerate(ctx.input_tensors):
            if x is None:
                raise ValueError(
                    f"Found None in ctx.input_tensors at index {i}! "
                    f"可能原因：forward 的第 {i} 个输入未正确传递或保存。"
                )

        if hasattr(ctx, "training_state"):
            for name, module in ctx.run_function.named_modules():
                if name in ctx.training_state:
                    module.training = ctx.training_state[name]
        
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        
        with torch.enable_grad():

            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)



def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)



class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)



def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)




def conv_nd_trans(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif dims == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif dims == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")




# cross att #
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        axial = axial.reshape(-1, t, d)

        axial = self.fn(axial, **kwargs)

        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


# axial pos emb
class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x




# save log ckpt
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)



def save_logs_to_json(log, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    log_dict = {
        key: value.item() if isinstance(value, torch.Tensor) and value.numel() == 1
              else value.tolist() if isinstance(value, torch.Tensor)
              else value
        for key, value in log.items()
    }

    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                all_logs = json.load(f) 
            except json.JSONDecodeError:
                all_logs = [] 
        all_logs = []

    all_logs.append(log_dict)  

    with open(path, 'w') as f:
        json.dump(all_logs, f, indent=4)



# diffusion  #

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()



def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out



def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev



def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)



def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module



def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()




