from config import *
import utils
from train import train
import numpy as np
import math
from evaluate import evaluate,evaluate_N,try_output
from sklearn.model_selection import KFold
import jax.numpy as jnp

X_raw,Y_raw=utils.Data_Loader(X_path,Y_path)
Y0=Y_raw
Y_delta=Y_raw-X_raw
print(X_raw[1000:1005],Y_delta[1000:1005])
Standard=utils.Standarder()

X=Standard.fit_transform(X_raw)
Y=Standard.transform_delta(Y_delta)

def Train_model(X,Y,choice1,choice2):

    if(choice2=="1"):

        params,model,loss=train(
            X,Y,
            model_name="MLP",
            output_dim=Y.shape[-1],
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            lr_start=Lr_start_MLP,
            lr_end=Lr_end_MLP,
            lrE=LrE,
            lrL=LrL,
            weight_decay_rate=WEIGHT_DECAY_RATE,
            X_scaler=None,
            Y_scaler=None,
            num_epoches=NUM_EPOCHES,
            seed=SEED)
        
    if(choice2=="2"):

        params,model,loss=train(
            X,Y,
            model_name="EGNN",
            output_dim=Y.shape[-1],
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            lr_start=Lr_start_EGNN,
            lr_end=Lr_end_EGNN,
            lrE=LrE,
            lrL=LrL,
            weight_decay_rate=WEIGHT_DECAY_RATE,
            X_scaler=None,
            Y_scaler=None,
            num_epoches=NUM_EPOCHES,
            seed=SEED)
    
    if(choice1=="1"):

        print("训练完毕！")

        if(choice2=="1"):
            utils.save_params_npz(params,MLP_weight_path)
        if(choice2=="2"):
            utils.save_params_npz(params,EGNN_weight_path)

        utils.plot_loss(loss)

    if(choice1=="3"):

        if(choice2=="1"):
            utils.save_params_npz(params,MLP_weight_path)
        if(choice2=="2"):
            utils.save_params_npz(params,EGNN_weight_path)


def Eval_model(X,Y,choice1,choice2):

    if(choice2=="1"):

        params=utils.load_params_npz(MLP_weight_path)
        loss=evaluate("MLP",params,X,Y,Y.shape[-1],HIDDEN_DIM,BATCH_SIZE,seed=SEED,is_training=False)
        return loss

    if(choice2=="2"):

        params=utils.load_params_npz(EGNN_weight_path)
        loss=evaluate("EGNN",params,X,Y,Y.shape[-1],HIDDEN_DIM,BATCH_SIZE,seed=SEED,is_training=False)
        return loss
    
def Eval_model_N(X,choice2):

    if(choice2=="1"):
        params=utils.load_params_npz(MLP_weight_path)
        X_list=evaluate_N("MLP",params,X,X.shape[-1],HIDDEN_DIM,num_iter=100,seed=SEED_Eva,is_training=False)
        return X_list

    if(choice2=="2"):
        params=utils.load_params_npz(EGNN_weight_path)
        X_list=evaluate_N("EGNN",params,X,X.shape[-1],HIDDEN_DIM,num_iter=100,seed=SEED_Eva,is_training=False)
        return X_list

if __name__=="__main__":

    choice1=input("选择模式：1.训练 2.评估 3. 5折交叉验证 4.N次连续评估 5.验证下一步")
    choice2=input("选择模型：1.MLP 2.EGNN")

    if choice1=="1":
        Train_model(X,Y,choice1,choice2)

    if choice1=="2":
        loss=Eval_model(X,Y,choice1,choice2)
        print(f"评估误差为{loss}")

    if choice1=="3":

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Fold {fold+1}/5")

            # 划分训练和验证
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            # 训练模型（每折重新初始化）
            model = Train_model(X_train, Y_train,choice1,choice2)

            # 验证模型
            val_loss = Eval_model(X_val, Y_val,choice1,choice2)
            print(f"Fold {fold+1}验证 loss: {val_loss:.6f}")

            fold_results.append(val_loss)

        # 输出 5 折平均指标
        mean_loss = np.mean(fold_results)
        print(f"平均验证loss: {mean_loss:.6f}")

    if choice1=="4":
        
        X_list=Eval_model_N(X,choice2)
        X_list=jnp.array(X_list)
        X_list=X_list.reshape((X_list.shape[0],X_list.shape[-1]))
        X_list=Standard.inverse_transform(X_list)
        X_list=np.array(X_list)

        print(f"前五行示例：")
        print(X_list[:5])

        if(choice2=="1"):
            np.save(MLP_evaluate_path,X_list)

        if(choice2=="2"):
            np.save(EGNN_evaluate_path,X_list)

    if choice1=="5":

        if choice2=="1":
            params=utils.load_params_npz(MLP_weight_path)
            try_output("MLP",params,X,Y0,X.shape[-1],HIDDEN_DIM,Standard,seed=SEED,is_training=False)

        if choice2=="2":
            params=utils.load_params_npz(EGNN_weight_path)
            try_output("EGNN",params,X,Y0,X.shape[-1],HIDDEN_DIM,Standard,seed=SEED,is_training=False)
