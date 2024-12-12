"""Evaluate different player names against each other."""

if __name__ == '__main__':
  import itertools
  import os
  import time
  import typing as tp

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin
  from slippi_ai import evaluators, flag_utils

  dolphin_config = dolphin.DolphinConfig(
      headless=True,
      infinite_time=True,
      path=os.environ.get('DOLPHIN_PATH'),
      iso=os.environ.get('ISO_PATH'),
  )
  DOLPHIN = ff.DEFINE_dict(
      'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')
  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', True, 'Use async environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')
  INNER_BATCH_SIZE = flags.DEFINE_integer(
      'inner_batch_size', 1, 'Number of environments to run sequentially.')

  AGENT_FLAGS = eval_lib.BATCH_AGENT_FLAGS.copy()
  del AGENT_FLAGS['name']
  AGENT_FLAGS['num_names'] = ff.Integer(4, 'Number of names to evaluate.')
  AGENT_FLAGS['names'] = ff.StringList(None)

  USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')

  AGENT = ff.DEFINE_dict('agent', **AGENT_FLAGS)

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')


  def main(_):

    agent_kwargs: dict = AGENT.value.copy()
    state = eval_lib.load_state(
        path=agent_kwargs.pop('path'),
        tag=agent_kwargs.pop('tag'))
    agent_kwargs.update(
        state=state,
        batch_steps=NUM_AGENT_STEPS.value,
    )

    num_names: int = agent_kwargs.pop('num_names')
    names: tp.Optional[list[str]] = agent_kwargs.pop('names')
    name_map: dict[str, int] = state['name_map']
    if names is None:
      code_to_name = {v: k for k, v in name_map.items()}
      names = [code_to_name[i] for i in range(num_names)]
    else:
      for name in names:
        if name not in name_map:
          raise ValueError(f'Invalid name "{name}"')
    print('Evaluating names:', names)

    per_name_kwargs: dict[str, dict] = {}
    for name in names:
      per_name_kwargs[name] = agent_kwargs.copy()
      per_name_kwargs[name].update(name=name)

    players = {
        1: dolphin.AI(),
        2: dolphin.AI(),
    }
    env_kwargs = dict(
        players=players,
        **dolphin.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
    )

    matchups = list(itertools.combinations(names, 2))

    results = []

    start_time = time.perf_counter()
    for i, (name1, name2) in enumerate(matchups):
      print(f'Evaluating "{name1}" vs "{name2}"')
      evaluator = evaluators.Evaluator(
          agent_kwargs={1: per_name_kwargs[name1], 2: per_name_kwargs[name2]},
          dolphin_kwargs=env_kwargs,
          num_envs=NUM_ENVS.value,
          async_envs=ASYNC_ENVS.value,
          env_kwargs=dict(
              num_steps=NUM_ENV_STEPS.value,
              inner_batch_size=INNER_BATCH_SIZE.value,
          ),
          use_gpu=USE_GPU.value,
      )

      with evaluator.run():
        metrics, timings = evaluator.rollout(ROLLOUT_LENGTH.value)
      del timings
      # print(timings)

      total_reward = metrics[1].reward
      num_frames = ROLLOUT_LENGTH.value * NUM_ENVS.value
      mean_reward = total_reward / num_frames
      reward_per_minute = mean_reward * 60 * 60

      print(f'Reward per minute: {reward_per_minute:.2f}')
      results.append((name1, name2, reward_per_minute))

      elapsed_time = time.perf_counter() - start_time
      n = i + 1
      time_per_item = elapsed_time / n
      time_left = time_per_item * (len(matchups) - n)
      print(f'Estimated time left: {time_left:.0f}')

    print('Results:')
    for name1, name2, reward in results:
      print(f'"{name1}" vs "{name2}": {reward:.2f}')

  app.run(main)
