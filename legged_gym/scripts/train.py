# 这一行代码导入很重要，因为在 `legged_gym.envs` 中的 `__init__.py` 文件中注册了任务环境
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    # returns `env` and `env_cfg`
    env, _ = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args
    )
    # TODO
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )


if __name__ == '__main__':
    args = get_args()
    train(args)
