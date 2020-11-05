""" 
This script runs the RL experiment in the Adam project of group 4 of STOR893.

The experiment is run in python 3.8 with these packages:
numpy=1.12.1, tensorflow==1.14.0, tflearn=0.3.2, gym=0.9.1, mujoco-py==0.5.7.

Note that mujoco-py CANNOT be simply installed by importing given environment. Please refer to http://www.mujoco.org/ for installing mujoco.
For windows user, please also refer to this trouble shooting: https://stackoverflow.com/questions/38766267/python-binding-for-mujoco-physics-library-using-mujoco-py-package
Even with the same random seeds, the result is not reproducible for GPU machine due to non-deterministicity of tensorflor-gpu.

Credits:
The implementation is largely based on a nice DDPG implementation from this paper: Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam
Git: https://github.com/emtiyaz/vadam/tree/master/tensorflow_RL

"""

#to suppress tensorflow system output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# argparse part 
import argparse
import time

import tensorflow as tf
import numpy as np
import gym
from Replay_buffer import Replay_buffer
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# ===========================
#   Actor and Critic DNNs
# ===========================
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def run(args):
    tf.reset_default_graph()

    savepath = './result/' + (args.env_name) + '/' + (args.method_name.upper())
        
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    Result_Path = savepath + ("/%s_lr%s_seed%s" % \
            (args.method_name.lower(), str(args.actor_lr), str(args.seed)  ) )

    print(Result_Path)

    # Directory for storing tensorboard summary results
    SUMMARY_DIR = savepath + '/tf_%s' % args.env_name

    # Size of replay buffer
    BUFFER_SIZE = 1000000
    MINIBATCH_SIZE = 64
    TEST_EPISODES = 20
    TEST_EVERY = 10
    n_layer1 = 400
    n_layer2 = 300


    with tf.Session() as sess:
        env = gym.make(args.env_name)

        # set random seeds
        # Note that even if we have set up the seeds, the result may still be different while reproducing it.
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        env.seed(args.seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high


        # actor - policy network; critic - Q network
        actor = ActorNetwork(sess=sess, state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, method_name=args.method_name, learning_rate=args.actor_lr, tau=args.tau)
        critic = CriticNetwork(sess, state_dim, action_dim, args.critic_lr, args.tau, actor.get_num_trainable_vars())

        # Set up summary Ops
        summary_ops, summary_vars = build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

        # Initialize replay memory
        replay_buffer = Replay_buffer(BUFFER_SIZE, args.seed)
        episode_reward = []
        episode_q = []

        test_reward = np.zeros((args.max_episode//TEST_EVERY,TEST_EPISODES))

        # We use double loop of max_episode and max_episode_timestep here.
        # For tasks with varied episode length, we need different types of loop for evaluation.
        for i in range(args.max_episode):

            s = env.reset()

            ep_reward = 0
            ep_ave_max_q = 0
            ep_ave_q = []

            for j in range(args.max_step):

                if args.render:
                    env.render()

                """ Compute action """
                a = actor.predict(s[np.newaxis,:])

                """ Take a step """
                s2, r, terminal, info = env.step(a[0])

                """ No future for final time step """
                if j == args.max_step - 1 :
                    terminal = 1
                    
                replay_buffer.store_transition(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r, terminal, np.reshape(s2, (actor.s_dim,)))

                """ Start training when we have at least M minibatch size samples """
                if replay_buffer.size() > MINIBATCH_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)   # state, action, reward, terminal, next_state

                    """ Calculate targets based on the action from the target actor """
                    target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                    """ Cut trace of Q for terminal state"""
                    y_i = []
                    for k in range(MINIBATCH_SIZE):
                       if t_batch[k]:
                           y_i.append(r_batch[k])
                       else:
                           y_i.append(r_batch[k] + args.gamma_rl * target_q[k])

                    """ Update the critic given the targets """
                    predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                    """ recording Q-value for checking and debugging """
                    ep_ave_max_q += np.amax(predicted_q_value)
                    ep_ave_q += [np.mean(predicted_q_value)]

                    """ No parameter exploration """
                    a_outs = actor.predict(s_batch)
                    grads = critic.action_gradients(s_batch, a_outs)

                    actor.train(s_batch, grads[0])

                    """ Update target networks """
                    actor.update_target_network()
                    critic.update_target_network()

                s = s2
                ep_reward += r

                """ Print statistics at the end of an episode """
                if terminal:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: ep_reward,
                        summary_vars[1]: ep_ave_max_q / float(j)
                    })
                    writer.add_summary(summary_str, i)
                    writer.flush()

                    result_text = "[" + time.strftime('%H:%M:%S') + "] | "
                    result_text += "Episode {}: Train return {:.4f}".format(i, ep_reward)

                    print(result_text)

                    break

            episode_reward += [ep_reward]
            episode_q += [np.mean(ep_ave_q)]

            if int(i+1) %100 == 0:
                print('Average reward in the first {} episodes is {:.4f}.'.format(i, np.mean(np.array(episode_reward))))

                path = Result_Path + '_train_reward.npy'
                np.save(path, episode_reward)

            if int(i+1) %1000 == 0:
                """ Save the model every X episodes """
                saver = tf.train.Saver()
                model_path = Result_Path + '_model'
                saver.save(sess, model_path, global_step=i)

            if i % TEST_EVERY == 0:
                print('Testing the actor network.......')
                actor.update_test_network() # Update the test network by hard copy.

                test_r_array = []
                for T in range(TEST_EPISODES):
                    test_r = 0
                    test_s = env.reset()
                    for j in range(args.max_step):
                        a = actor.predict_test(test_s[np.newaxis,:])
                        test_s2, r, terminal, info = env.step(a[0])
                        test_s = test_s2
                        test_r += r
                        if terminal:
                            break
                    test_reward[i//TEST_EVERY, T] = test_r
                    test_r_array += [test_r]
                test_r_array = np.array(test_r_array)
                result_text = "--Episode {}: Test return: {:.4f}({:.4f})".format(i, np.mean(test_r_array), np.std(test_r_array) / np.sqrt(TEST_EPISODES))

                print(result_text)
                    
                test_path = Result_Path + '_test_reward.npy'
                np.save(test_path, test_reward)
                
                test_path_text = Result_Path + '_test_reward.txt'
                with open(test_path_text, 'a') as f:
                    print(result_text, file=f) 


        plt.figure()
        plt.plot(episode_reward)
        plt.xlabel('Episode')
        plt.ylabel('Train Rewards')
        plotpath = Result_Path +  '_train_reward.pdf'
        plt.savefig(plotpath)

        plt.figure()
        mean_test_reward = np.mean(test_reward, axis=1)
        plt.plot(mean_test_reward)
        plt.xlabel('Episode')
        plt.ylabel('Test Rewards')
        plotpath = Result_Path +  '_test_reward.pdf'
        plt.savefig(plotpath)

        sess.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run experiments.")

    ## Environment and experiment options
    parser.add_argument('--env_name', help='name of the environment to run')
    '''
    env_id modified here
    '''
    parser.add_argument("--env_id", dest="env_id", default=0, type=int, help="Index to environment. Ignored if env_name is provided.")
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument("--max_episode", default=700, type=int, help="Maximum episodes")
    parser.add_argument("--max_step", default=1000, type=int, help="Maximum steps in an episode")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    ## critic update options
    parser.add_argument("--critic_lr", default=0.0025, type=float, help="Critic learning rate (omega), 1e-3 ")
    parser.add_argument("--gamma_rl", default=0.99, type=float, help="Discount factor")
    parser.add_argument("--tau", default=0.001, type=float, help="Soft target update param")

    ## Choose the actor optimization method
    ## name in the paper: "noise+sgd" is SGD-Explore, "noise+adam" is Adam-Explore, "sgd" is SGD-Plain, and "adam" is Adam-Plain.
    parser.add_argument("--method_name", dest="method_name", default="adam", help="Available optimizer: adam, adagrad, sgd, rmsprop.")

    ## Actor learning options. Name in paranthesis refers to the paper and Table 3 in the paper.
    parser.add_argument("--actor_lr", default=0.0001, type=float, help="(alpha) Actor learning rate.")
    parser.add_argument("--gamma_2", default=0.999, type=float, help="(gamma_2) Second-order moment learning rate for Vadam, precision step size for VadaGrad, or variance step-size in noise+X.")
    parser.add_argument("--gamma_1", default=0.9, type=float, help="(gamma_1) First-order moment learning rate for Vadam.")
    parser.add_argument("--prior", default=1e-8, type=float, help="(lambda) Prior precision for Vadam.")
    parser.add_argument("--prec_init", default=0, type=int, help="(s_1) Initial value of the second-order moment estimation")
    parser.add_argument("--min_prec", default=10000, type=int, help="(c) Added constant to precision for Vadam")

    parser.add_argument("--actor_lr_sigma", default=0.01, type=float, help="(alpha^{sigma}) variance actor learning rate.")
    
    args = parser.parse_args()
        
    if args.env_name is None:
        env_dict = {
                    0 : "Pendulum-v0",
        }
        args.env_name = env_dict[args.env_id]

    run(args=args)

    