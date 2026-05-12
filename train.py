import jax
import jax.numpy as jnp
import optax
import haiku as hk
import math
from models import MLP_MAKER,EGNN_MAKER
from utils import Data_Iter,compute_physical_quantities

def train(X,Y,model_name,output_dim,hidden_dim,batch_size,lr_start=1e-3,lr_end=11e-5,lrE=1,lrL=1,weight_decay_rate=1e-4,X_scaler=None,Y_scaler=None,num_epoches=50,seed=42):

    key=jax.random.PRNGKey(seed)
    X=jnp.array(X)
    Y=jnp.array(Y)

    match model_name:
        case "MLP":
            model=MLP_MAKER(output_dim,hidden_dim)
        case "EGNN":
            model=EGNN_MAKER(16,0.01,6)
        case _:
            assert 1==0,"没有此模型！"

    steps_per_epoch = math.ceil(X.shape[0]/batch_size)
    transition_steps = max(1, num_epoches * steps_per_epoch)

    schedule = optax.linear_schedule(
    init_value=lr_start,
    end_value=lr_end,
    transition_steps=transition_steps
    )



    params=model.init(key,jnp.zeros((1,X.shape[-1])),is_training=True)
    optimizer = optax.chain(
    optax.add_decayed_weights(weight_decay_rate), 
    optax.adam(schedule)
    )
    opt_state=optimizer.init(params)

    @jax.jit(static_argnames=['is_training','X_scaler','Y_scaler'])
    def loss(params,x,y,key,is_training=True,lrE=1.0,lrL=1.0,X_scaler=None,Y_scaler=None):
        y_hat=model.apply(params,key,x,is_training=is_training)

        #无物理学约束
        mse_loss=jnp.mean((y_hat-y)**2)
   
        # 如果没有提供 scaler 或者不是训练模式，不计算物理损失
        if X_scaler is None or Y_scaler is None or not is_training:
            return mse_loss

        # 反归一化到物理量
        y_hat_phys = Y_scaler.inverse_transform(y_hat)
        y_phys = Y_scaler.inverse_transform(y)

        # 计算能量和角动量
        E_hat, L_hat = compute_physical_quantities(y_hat_phys, Y_scaler)
        E_true, L_true = compute_physical_quantities(y_phys, Y_scaler)

         # 物理约束损失
        phys_loss1 = lrE * jnp.mean((E_hat - E_true)**2)
        phys_loss2= lrL * jnp.mean((L_hat - L_true)**2)

        loss = mse_loss + phys_loss1+phys_loss2
        print(f"M:{mse_loss[0,1]},E:{phys_loss1[0,1]},L:{phys_loss2[0,1]}")

        return loss

    @jax.jit(static_argnames=['is_training','X_scaler','Y_scaler'])
    def update(params,opt_state,x,y,key,is_training=True,lrE=lrE,lrL=lrL,X_scaler=X_scaler,Y_scaler=Y_scaler):
        grads=jax.grad(loss)(params,x,y,key,is_training,lrE=lrE,lrL=lrL,X_scaler=X_scaler,Y_scaler=Y_scaler)
        updates,opt_state=optimizer.update(grads,opt_state,params=params)
        params=optax.apply_updates(params,updates)
        return params,opt_state

    epoch_loss=[]

    for epoch in range(num_epoches):

        key,subkey=jax.random.split(key)

        for X_batch,Y_batch in Data_Iter(X,Y,batch_size,subkey):
            subkey,subsubkey=jax.random.split(subkey)
            params,opt_state=update(params,opt_state,X_batch,Y_batch,subsubkey,is_training=True,lrE=lrE,lrL=lrL,X_scaler=X_scaler,Y_scaler=Y_scaler)

        epoch_loss_=loss(params,X,Y,key=None,is_training=False)
        print(f"Epoch:{epoch+1}/{num_epoches},Loss:{epoch_loss_:.6f}")
        epoch_loss.append(epoch_loss_)
    
    return params,model,epoch_loss

