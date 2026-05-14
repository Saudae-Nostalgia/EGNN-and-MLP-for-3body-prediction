import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
from utils import Trans,Inverse_Trans
import numpy as np

def MLP_MAKER(output_dim,hidden_num=256):

    def forward(X,is_training):
   

        X=hk.Linear(hidden_num)(X)
        X=jax.nn.silu(X)

        X0=X 
        X=hk.Linear(hidden_num)(X)
        X=jax.nn.silu(X)
        X = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(X)
        X = X + X0
    
        X=hk.Linear(hidden_num)(X)
        X=jax.nn.silu(X)

        X0=X
        X = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(X)
        X=hk.Linear(hidden_num)(X)
        X=jax.nn.silu(X)
        X = X + X0
        
        X=hk.Linear(hidden_num)(X)
        X=jax.nn.silu(X)

        X=hk.Linear(output_dim-3)(X)
        mass_delta = jnp.zeros((X.shape[0], 3), dtype=X.dtype)
        X = jnp.concatenate([mass_delta, X], axis=-1)

        return X
    
    return hk.transform(forward)

def EGNN_MAKER(hidden_num=16,init_stddev=0.01,depth=3):
    

    def phie(x, h, d):

        n, dim = x.shape 
        rij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        rij = jnp.sum(jnp.square(rij), axis=-1).reshape(n, n, 1)

        mlp = hk.nets.MLP([hidden_num, hidden_num], w_init=hk.initializers.TruncatedNormal(init_stddev), activation=jax.nn.silu, name=f"edge_mlp_{d}") 
        @partial(hk.vmap, in_axes=(0, None, 0), out_axes=0, split_rng=False)
        @partial(hk.vmap, in_axes=(None, 0, 0), out_axes=0, split_rng=False)
        def phi(hi, hj, r):
            hhr = jnp.concatenate([hi, hj, r], axis=0)
            return mlp(hhr)
        return phi(h, h, rij)


    def phix(mij, d, is_training):
        mlp = hk.nets.MLP([hidden_num, 1], w_init=hk.initializers.TruncatedNormal(init_stddev), activation=jax.nn.silu, name=f"coord_mlp_{d}")
        
        mij=mlp(mij)

        return mij


    def phih(h, m, d, is_training):
        hm = jnp.concatenate([h, m], axis=-1)
        mlp = hk.nets.MLP([hidden_num, h.shape[-1]], w_init=hk.initializers.TruncatedNormal(init_stddev), activation=jax.nn.silu, name=f"node_mlp_{d}")
        hm=mlp(hm)

        return hm+h


    def block(x,h,d, is_training):
            
            n,dim=x.shape

            alpha=1.0

            mij = phie(x, h, d)

            xij = jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim))

            mask = ~np.eye(n, dtype=bool) # maskout diagonal
            mij = mij[mask].reshape(n, n-1, hidden_num)
            xij = xij[mask].reshape(n, n-1, dim)

            weight = phix(mij, d, is_training).reshape(n, n-1)/(n-1)

            x = x + alpha * jnp.einsum('ijd,ij->id', xij, weight)

            m = jnp.sum(mij, axis=1)

            h = phih(h, m, d, is_training) 
            return x, h 
    
    def forward(X,is_training):

        x,h=Trans(X)
        batch_block=jax.vmap(block,in_axes=(0,0,None,None),out_axes=(0,0))

        for d in range(depth):
            x, h = batch_block(x,h,d,is_training)

        X=Inverse_Trans(x,h)
        
        return X

    return hk.transform(forward)
    



