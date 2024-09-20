
from agent.trainer.ppo_trainer import PPOTrainer
from absl import flags, app
from agent.environments import suite_gibson
import os

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the gin config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Gin binding to pass through.')

flags.DEFINE_integer('num_iterations', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of steps to collect at the beginning of training using random policy')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of steps to collect and be added to the replay buffer after every training iteration')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_parallel_environments_eval', 1,
                     'Number of environments to run in parallel for eval')
flags.DEFINE_integer('replay_buffer_capacity', 1000000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of training steps in every training iteration')
flags.DEFINE_integer('batch_size', 256,
                     'Batch size for each training step. '
                     'For each training iteration, we first collect collect_steps_per_iteration steps to the '
                     'replay buffer. Then we sample batch_size steps from the replay buffer and train the model'
                     'for train_steps_per_iteration times.')
flags.DEFINE_float('gamma', 0.99,
                   'Discount_factor for the environment')
flags.DEFINE_float('actor_learning_rate', 3e-4,
                   'Actor learning rate')
flags.DEFINE_float('critic_learning_rate', 3e-4,
                   'Critic learning rate')
flags.DEFINE_float('alpha_learning_rate', 3e-4,
                   'Alpha learning rate')

flags.DEFINE_integer('num_eval_episodes', 10,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('eval_interval', 10000,
                     'Run eval every eval_interval train steps')
flags.DEFINE_boolean('eval_only', False,
                     'Whether to run evaluation only on trained checkpoints')
flags.DEFINE_boolean('eval_deterministic', False,
                     'Whether to run evaluation using a deterministic policy')
flags.DEFINE_integer('gpu_c', 1,
                     'GPU id for compute, e.g. Tensorflow.')

# Added for Gibson
flags.DEFINE_string('config_file', None,
                    'Config file for the experiment.')
flags.DEFINE_string('agent_config_file', None,
                    'Config file for the agent.')
flags.DEFINE_list('model_ids', None,
                  'A comma-separated list of model ids to overwrite config_file.'
                  'len(model_ids) == num_parallel_environments')
flags.DEFINE_list('model_ids_eval', None,
                  'A comma-separated list of model ids to overwrite config_file for eval.'
                  'len(model_ids) == num_parallel_environments_eval')
flags.DEFINE_string('env_mode', 'headless',
                    'Mode for the simulator (gui or headless)')
flags.DEFINE_float('action_timestep', 1.0 / 10.0,
                   'Action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 40.0,
                   'Physics timestep for the simulator')
flags.DEFINE_integer('gpu_g', 1,
                     'GPU id for graphics, e.g. Gibson.')
flags.DEFINE_boolean('random_position', False,
                     'Whether to randomize initial and target position')

flags.DEFINE_boolean('generate_data', False,
                     'generate data')

flags.DEFINE_integer('num_episodes', 5,
                     'the amount of episodes')

FLAGS = flags.FLAGS


def main(argv):
    FLAGS(argv)
    trainer = PPOTrainer(FLAGS)

    if FLAGS.generate_data == True:
        trainer.generate_data(
            env_load_fn=lambda model_id, mode, device_idx: suite_gibson.load(
                config_file=FLAGS.config_file,
                model_id=model_id,
                env_mode=mode,
                action_timestep=FLAGS.action_timestep,
                physics_timestep=FLAGS.physics_timestep,
                device_idx=device_idx,
            ),
            model_ids=FLAGS.model_ids,
            num_episodes=FLAGS.num_episodes,
        )

    elif FLAGS.eval_only == False:
        trainer.train(
            env_load_fn=lambda model_id, mode, device_idx: suite_gibson.load(
                config_file=FLAGS.config_file,
                model_id=model_id,
                env_mode=mode,
                action_timestep=FLAGS.action_timestep,
                physics_timestep=FLAGS.physics_timestep,
                device_idx=device_idx,
            ),
        )
    elif FLAGS.eval_only == True:
        trainer.eval(
            env_load_fn=lambda model_id, mode, device_idx: suite_gibson.load(
                config_file=FLAGS.config_file,
                model_id=model_id,
                env_mode=mode,
                action_timestep=FLAGS.action_timestep,
                physics_timestep=FLAGS.physics_timestep,
                device_idx=device_idx,
            ),
            model_ids=FLAGS.model_ids
        )

if __name__ == '__main__':
    
    app.run(main)

