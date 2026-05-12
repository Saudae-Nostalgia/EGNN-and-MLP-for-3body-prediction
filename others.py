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
    X,Y=utils.Data_Loader(X_path_0,Y_path_0)

    X=np.array(X)
    Y=np.array(Y)

    X_m=utils.mirror_data(X,axis='x')
    Y_m=utils.mirror_data(Y,axis='x')

    X_i=utils.rotate_data(X,math.pi)
    Y_i=utils.rotate_data(Y,math.pi)

    X_im=utils.mirror_data(X_i,axis='x')
    Y_im=utils.mirror_data(Y_i,axis='x')

    X=np.concatenate([X,X_m,X_i,X_im],axis=0)
    Y=np.concatenate([Y,Y_m,Y_i,Y_im],axis=0)

    print(f"X:{X.shape},Y:{Y.shape}")

    os.makedirs("data", exist_ok=True)
    np.save(os.path.join("data", "X_new.npy"), X)
    np.save(os.path.join("data", "Y_new.npy"), Y)

def test1():
    """
    该函数用于验证归一化函数及其反函数是否起效

    """
    X,Y=utils.Data_Loader(X_path,Y_path)

    print(X[5])

    X_standard=utils.Standarder()
  
    X_norm=X_standard.fit_transform(X)

    print(X_norm[5])
    X_result=X_standard.inverse_transform(X_norm)
    print(X_result[5])


def test2():
     """
    该函数用于将原始数据转换为EGNN所需格式数据的函数及其反函数是否有效

    """
     X,Y=utils.Data_Loader(X_path,Y_path)
     print(X[5])
     x,h=Trans(X)
     X=Inverse_Trans(x,h)
     print(X[5])

data_maker()