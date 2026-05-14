import jax.numpy as jnp
import jax
import numpy as np
import math
import matplotlib.pyplot as plt
from config import *
from jax import tree_util

def Data_Loader(X_path,Y_path):

    X=np.load(X_path)
    Y=np.load(Y_path)

    print(f"X:{X.shape},Y:{Y.shape}")

    X=jnp.array(X)
    Y=jnp.array(Y)

    return X,Y
"""
def save_params_npz(params, filepath=weight_path):
    # 把 DeviceArray 转成 NumPy
    params_np = tree_util.tree_map(lambda x: np.array(x), params)
    # flatten tree 为字典路径形式
    flat_params = {}
    def flatten_dict(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                flatten_dict(v, key)
            else:
                flat_params[key] = v
    flatten_dict(params_np)
    np.savez(filepath, **flat_params)

def load_params_npz(filepath=weight_path):
    data = np.load(filepath)
    # 还原嵌套 dict（如果你想恢复原始 tree）
    params = {}
    for k in data:
        # 这里用 "/" 分割恢复层级
        keys = k.split("/")
        d = params
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = jnp.array(data[k])
    return params

"""
def save_params_npz(params, filepath=weight_path):
    # Haiku params 结构严格为两层： {module_name: {param_name: array}}
    params_np = tree_util.tree_map(lambda x: np.array(x), params)
    flat_params = {}
    
    for module_name, module_dict in params_np.items():
        for param_name, value in module_dict.items():
            # 用 / 拼接，完美展平
            key = f"{module_name}/{param_name}"
            flat_params[key] = value
            
    np.savez(filepath, **flat_params)


def load_params_npz(filepath=weight_path):
    data = np.load(filepath)
    params = {}
    
    for k in data:
        # 关键修复：使用 rsplit("/", 1) 从右侧分割一次
        # 这样 "edge_mlp_0/~/linear_0/w" 就会被安全切分为 ["edge_mlp_0/~/linear_0", "w"]
        parts = k.rsplit("/", 1)
        if len(parts) == 2:
            module_name, param_name = parts
            if module_name not in params:
                params[module_name] = {}
            params[module_name][param_name] = jnp.array(data[k])
            
    return params


def Data_Iter(X,Y,batch_size,key):

    assert X.shape[0]==Y.shape[0],f"样本数量不一致！X:{X.shape[0]},Y:{Y.shape[0]}"

    num_samples=X.shape[0]
    indices=jnp.arange(num_samples)
    indices=jax.random.permutation(key,num_samples)
    
    for start in range(0,num_samples,batch_size):
        end=start+batch_size
        batch_idx=indices[start:end]
        yield X[batch_idx],Y[batch_idx]

class Standarder:
    def __init__(self,eps=1e-8):
        self.eps=eps
        self.mean_=None
        self.std_=None

    def fit(self,X):
        self.mean_=jnp.mean(X, axis=0)
        self.std_=jnp.std(X, axis=0)
        return self

    def transform(self,X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("尚未拟合")
        return (X-self.mean_)/(self.std_+self.eps)

    def transform_delta(self,delta):
        if self.std_ is None:
            raise ValueError("Standarder has not been fitted")
            raise ValueError("灏氭湭鎷熷悎")
        return delta/(self.std_+self.eps)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,X_scaled):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("尚未拟合")
        return X_scaled * (self.std_ + self.eps) + self.mean_

    def inverse_transform_delta(self,delta_scaled):
        if self.std_ is None:
            raise ValueError("Standarder has not been fitted")
            raise ValueError("灏氭湭鎷熷悎")
        return delta_scaled * (self.std_ + self.eps)
    
def rotate_data(data, theta):

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    data_rot = data.copy()

    # 行星1位置
    x1, y1 = data[:,3], data[:,4]
    data_rot[:,3] = cos_t * x1 - sin_t * y1
    data_rot[:,4] = sin_t * x1 + cos_t * y1

    # 行星1速度
    vx1, vy1 = data[:,5], data[:,6]
    data_rot[:,5] = cos_t * vx1 - sin_t * vy1
    data_rot[:,6] = sin_t * vx1 + cos_t * vy1

    # 行星2位置
    x2, y2 = data[:,7], data[:,8]
    data_rot[:,7] = cos_t * x2 - sin_t * y2
    data_rot[:,8] = sin_t * x2 + cos_t * y2

    # 行星2速度
    vx2, vy2 = data[:,9], data[:,10]
    data_rot[:,9]  = cos_t * vx2 - sin_t * vy2
    data_rot[:,10] = sin_t * vx2 + cos_t * vy2

    return data_rot


def mirror_data(data, axis='x'):

    data_mirror = data.copy()

    if axis == 'x':
        data_mirror[:,4]  = -data_mirror[:,4]
        data_mirror[:,6]  = -data_mirror[:,6]
        data_mirror[:,8]  = -data_mirror[:,8]
        data_mirror[:,10] = -data_mirror[:,10]
    elif axis == 'y':
        data_mirror[:,3] = -data_mirror[:,3]
        data_mirror[:,5] = -data_mirror[:,5]
        data_mirror[:,7] = -data_mirror[:,7]
        data_mirror[:,9] = -data_mirror[:,9]
    else:
        raise ValueError("请输入正确轴")

    return data_mirror


def plot_loss(loss_list, title="Training Loss"):
    """
    绘制 loss 随 epoch 变化的折线图

    参数:
        loss_list: list 或 1D np.array, 每个元素对应一个 epoch 的 loss
        title: 图表标题
    """
    epochs = range(1, len(loss_list) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, loss_list, color='purple', marker='o', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_physical_quantities(X_norm, X_scaler):
    """
    计算每个样本的总能量和总角动量
    X_norm: jnp.array, shape=(batch_size, 11)，归一化后的数据
    X_scaler: 用于反归一化的 StandardScaler 对象

    返回:
        E: jnp.array, shape=(batch_size,) 总能量
        L: jnp.array, shape=(batch_size,) 总角动量
    """
    # 1. 反归一化回原始物理量
    X = X_scaler.inverse_transform(X_norm)

    # 质量
    m1 = X[:,0]
    m2 = X[:,1]
    m3 = X[:,2]

    # 行星1位置和速度
    x1 = X[:,3]; y1 = X[:,4]
    vx1 = X[:,5]; vy1 = X[:,6]

    # 行星2位置和速度
    x2 = X[:,7]; y2 = X[:,8]
    vx2 = X[:,9]; vy2 = X[:,10]

    # 为了方便，这里假设只有两颗行星（你描述的是三体？前三列质量，后面两个行星？）
    # 如果三体，需要再加第三个行星坐标/速度列
    
    # 速度平方
    v1_sq = vx1**2 + vy1**2
    v2_sq = vx2**2 + vy2**2

    # 动能 T = 1/2 m v^2
    T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq

    # 位矢差 r12 = r1 - r2
    dx = x1 - x2
    dy = y1 - y2
    r12 = jnp.sqrt(dx**2 + dy**2 + 1e-8)  # 避免除零

    # 引力常数 G 设为 1 (单位归一化)
    G = 1.0
    U = - G * m1 * m2 / r12  # 势能

    # 总能量
    E = T + U

    # 角动量 (二维平面，标量 L = r × p = m*(x vy - y vx))
    L1 = m1 * (x1 * vy1 - y1 * vx1)
    L2 = m2 * (x2 * vy2 - y2 * vx2)
    L = L1 + L2

    return E, L

def Trans(X):
        
        if X.ndim == 1: #防止单样本报错
            X = X[None, :]

        batch_size=X.shape[0]
        A_zeros = jnp.zeros((batch_size, 4))
        data_aug = jnp.concatenate([X[:, :3], A_zeros, X[:, 3:]], axis=1)
   
        x = jnp.stack([
            data_aug[:,3:5],      # A 位置
            data_aug[:,7:9],                # B 位置
            data_aug[:,11:13]               # C 位置
        ], axis=1)

        h = jnp.stack([
            jnp.stack([data_aug[:,0],  data_aug[:, 5], data_aug[:, 6]], axis=1),  # A: mass + vx,vy=0
            jnp.stack([data_aug[:, 1], data_aug[:, 9], data_aug[:, 10]], axis=1),           # B: mass, vx,vy
            jnp.stack([data_aug[:, 2], data_aug[:, 13], data_aug[:, 14]], axis=1)           # C: mass, vx,vy
        ], axis=1)

        return x,h

def Inverse_Trans(x, h):
        # A 星体质量
        A_mass = h[:,0,0]  # [B]
        # B 星体质量
        B_mass = h[:,1,0]
        # C 星体质量
        C_mass = h[:,2,0]

        # B 节点位置和速度
        B_x, B_y = x[:,1,0], x[:,1,1]
        B_vx, B_vy = h[:,1,1], h[:,1,2]

        # C 节点位置和速度
        C_x, C_y = x[:,2,0], x[:,2,1]
        C_vx, C_vy = h[:,2,1], h[:,2,2]

        # 拼接成原始 X [B,11]
        X = jnp.stack([
            A_mass, B_mass, C_mass,
            B_x, B_y, B_vx, B_vy,
            C_x, C_y, C_vx, C_vy
        ], axis=1)

        return X

def prepare_animation_data(X, dt=0.1):
    """
    将 X[N,11] 转为动画可用数据
    
    输入:
        X: np.ndarray, shape [N,11]
        dt: 时间步长，默认0.1
    
    输出:
        r: np.ndarray, shape [N,2], B星体位置
        rj: np.ndarray, shape [N,2], C星体位置
        t: np.ndarray, shape [N], 时间序列
    """
    N = X.shape[0]
    
    # B星体位置 xy
    r = X[:, 3:5]  # 列 3,4
    
    # C星体位置 xy
    rj = X[:, 7:9]  # 列 7,8
    
    # 时间序列
    t = np.arange(N) * dt  # [0, dt, 2*dt, ..., (N-1)*dt]
    
    return r, rj, t
