import optuna
from train import train, test  

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    layer_size = trial.suggest_int('layer_size', 10, 3000)

    model = train(lr, layer_size)  # You might need to change how you call this depending on how it's defined in train.py
    train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, 
                                         global_step, warmup_iters, writer, logging,
                                        )
    valid_neg_log_p, valid_nelbo = test(valid_queue, model, num_samples=10, args=args, logging=logging)

    return valid_neg_log_p

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print(study.best_trial.params)
