import jax
import jax.numpy as jnp
from models import MLP_MAKER,EGNN_MAKER
from config import *
def evaluate(model_name,params,X,Y,output_dim,hidden_dim,batch_size=256,seed=42,is_training=False):

    assert X.shape[0]==Y.shape[0],"数据维度不匹配！"
    num_samples=X.shape[0]
    mse_list=[]
    rng = jax.random.PRNGKey(seed)
    

    match model_name:
        case "MLP":
            model=MLP_MAKER(output_dim,hidden_dim)
        case "EGNN":
            model=EGNN_MAKER(16,0.01,6)
        case _:
            assert 1==0,"没有此模型！"

    params_0=model.init(rng,jnp.zeros((1,X.shape[-1])),is_training=False)

    @jax.jit(static_argnames=['is_training'])
    def forward(params,rng,x_batch,is_training=False):
        rng, rng_batch = jax.random.split(rng)
        y_hat=model.apply(params,rng_batch,x_batch,is_training)
        return y_hat
    
    for i in range(0,num_samples,batch_size):
        x_batch=X[i:i+batch_size]
        y_batch=Y[i:i+batch_size]
        y_hat = forward(params,rng, x_batch,is_training)
        mse_list.append(jnp.mean((y_hat - y_batch)**2))

    loss = float(jnp.mean(jnp.array(mse_list)))
    return loss

def evaluate_N(model_name,params,X,output_dim,hidden_dim,num_iter=1000,seed=42,is_training=False):

    num_samples=X.shape[0]
    X_list=[]
    rng = jax.random.PRNGKey(seed)
    idx=jax.random.randint(rng,(),0,num_samples)
    X_iter=X[idx]
    X_iter=X_iter[None, :]


    match model_name:
        case "MLP":
            model=MLP_MAKER(output_dim,hidden_dim)
        case "EGNN":
            model=EGNN_MAKER(16,0.01,6)
        case _:
            assert 1==0,"没有此模型！"

    params_0=model.init(rng,jnp.zeros((1,X.shape[-1])),is_training=False)

    @jax.jit(static_argnames=['is_training'])
    def forward(params,rng,x_batch,is_training=False):
        rng, rng_batch = jax.random.split(rng)
        y_hat=model.apply(params,rng_batch,x_batch,is_training)
        return y_hat
    
    for i in range(0,num_iter):    
        X_list.append(X_iter)
        X_iter0=X_iter
        X_iter = forward(params,rng, X_iter,is_training)
        X_iter = X_iter.at[0, :3].set(X_iter0[0, :3])

    return X_list