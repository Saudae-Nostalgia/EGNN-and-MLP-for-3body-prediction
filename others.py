from config import *
import utils
import numpy as np
import math
import os
import jax.numpy as jnp
from utils import Trans,Inverse_Trans

"""
此模块不属于本体,为一些制作过程中产生的一次性的小操作

"""

def data_maker():

    """
    该函数用于通过旋转、镜像数据来扩充数据集并且学习对称性

    """
    X,Y=utils.Data_Loader(X_path_0,Y_path_0)

    X=np.array(X)
    Y=np.array(Y)

    X_r1=utils.rotate_data(X,math.pi/4)
    X_r2=utils.rotate_data(X,math.pi/4*2)
    X_r3=utils.rotate_data(X,math.pi/4*3)
    X_r4=utils.rotate_data(X,math.pi/4*4)

    Y_r1=utils.rotate_data(Y,math.pi/4)
    Y_r2=utils.rotate_data(Y,math.pi/4*2)
    Y_r3=utils.rotate_data(Y,math.pi/4*3)
    Y_r4=utils.rotate_data(Y,math.pi/4*4)

    X=np.concatenate([X,X_r1,X_r2,X_r3,X_r4],axis=0)
    Y=np.concatenate([Y,Y_r1,Y_r2,Y_r3,Y_r4],axis=0)

    X_m1=utils.mirror_data(X,axis='x')
    X_m2=utils.mirror_data(X,axis='y')
    Y_m1=utils.mirror_data(Y,axis='x')
    Y_m2=utils.mirror_data(Y,axis='y')

    X=np.concatenate([X,X_m1,X_m2],axis=0)
    Y=np.concatenate([Y,Y_m1,Y_m2],axis=0)

    print(f"X:{X.shape},Y:{Y.shape}")
    datafolder = "data"

    os.makedirs(datafolder, exist_ok=True)

    np.save(os.path.join(datafolder, "X_new.npy"), X)
    np.save(os.path.join(datafolder, "Y_new.npy"), Y)

def test1():
    """
    该函数用于验证归一化函数及其反函数是否起效

    """
    X,Y=utils.Data_Loader(X_path,Y_path)

    print(X[5])

    X_standard=utils.Standarder()
    Y_standard=utils.Standarder()

    X_norm=X_standard.fit_transform(X)
    Y_norm=Y_standard.fit_transform(Y)

    print(X_norm[5])
    print(X_norm[500])
    print(X_norm[501])


def test2():
     """
    该函数用于将原始数据转换为EGNN所需格式数据的函数及其反函数是否有效

    """
     X,Y=utils.Data_Loader(X_path,Y_path)
     print(X[5])
     x,h=Trans(X)
     X=Inverse_Trans(x,h)
     print(X[5])
test2()