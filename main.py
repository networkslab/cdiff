from utils import get_args, run_train, run_eval

if __name__ == '__main__':
    args = get_args()
    args = run_train(args)
    args = run_eval(args)