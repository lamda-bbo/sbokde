from kdesbo.KDESBO import *
from Benchmark.Test_Function import *
import argparse
import botorch
import numpy as np
import random
import torch
import pickle as pkl
import wandb
import time
parser = argparse.ArgumentParser()
parser.add_argument('--Optimizer', default='SBOKDE', type=str)
# MMD / MMD_Minimax_Approx / SBOKDE / DRBOKDE / UCB / Stable
parser.add_argument('--TestProblem', default='Ackley', type=str)
# Ackley / Hartmann / Modified_Branin / portfolio_optimization
# / portfolio_normal_optimization / Hartmann_complicated / Continuous_Vendor
parser.add_argument('--Minimization', action="store_true")
parser.add_argument('--init_size', default=10, type=int)
parser.add_argument('--running_rounds', default=350, type=int)
parser.add_argument('--repeat', default=5, type=int)
parser.add_argument('--start_seed', default=100, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--beta', default=1.0, type=float)


args = parser.parse_args()

if (args.Optimizer=='MMD' or args.Optimizer=='DRBOKDE') and args.device!='cpu':
    raise ValueError(f"GPU of {args.Optimizer} is not supported.")
print(args)
device = torch.device(args.device)
print(device)
test_func = eval(args.TestProblem)(negate=not args.Minimization).to(dtype=torch.float64)
for i in range(args.repeat):
    wandb.init(
        project=f"SBOKDE, {args.TestProblem}, {args.running_rounds}, {test_func.dim}, {test_func.mu}, {test_func.sigma}",
        name=f"{args.Optimizer}_{args.start_seed+i}",
        config={
            "algorithm": f"{args.Optimizer}",
            "beta": f"{args.beta}"
        },
        reinit=True)
    start= time.time()
    seed = args.start_seed + i
    botorch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    opt = eval(args.Optimizer+'_Optimizer')(test_func, running_rounds=args.running_rounds, init_size=args.init_size, device=device, beta=args.beta)
    X, Y, contexts = opt.run_opt()
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_X.pkl','wb') as f:
            pkl.dump(X.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_Y.pkl','wb') as f:
            pkl.dump(Y.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_contexts.pkl','wb') as f:
            pkl.dump(contexts.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_stochastic_Y.pkl','wb') as f:
        if opt.stochastic_Y is not None:
            pkl.dump(opt.stochastic_Y.cpu().detach().numpy(), f)
        else:
            pkl.dump('', f)
    print(f"Time for {i}th run:{time.time() - start}")
    wandb.finish()
    torch.cuda.empty_cache()
