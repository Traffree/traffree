import numpy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

n_observations = 2  # 18
n_actions = 2


def create_tls_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n_observations, activation='relu'),
        # tf.keras.layers.Dense(units=n_hidden, activation='relu'),
        tf.keras.layers.Dense(units=n_actions, activation=None)
    ])
    return model


def choose_action(model, observation, single=True):
    # add batch dimension to the observation if only a single example was provided
    observation = np.expand_dims(observation, axis=0) if single else observation

    logits = model.predict(observation)
    action = tf.random.categorical(logits, num_samples=1)
    action = action.numpy().flatten()
    action = tf.one_hot(action, n_actions, axis=-1)

    return action[0] if single else action


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(new_reward)


# Helper function to combine a list of Memory objects into a single Memory.
#     This will be very useful for batching.
def aggregate_memories(memories):
    batch_memory = Memory()

    for memory in memories:
        for step in zip(memory.observations, memory.actions, memory.rewards):
            batch_memory.add_to_memory(*step)

    return batch_memory


def normalize(x):
    x -= np.mean(x)
    x /= np.std(x)
    return x.astype(np.float32)


def discount_rewards(rewards, gamma=0.95):
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        # update the total discounted reward
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R

    return normalize(discounted_rewards)


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


def train_step(model, optimizer, observations, actions, discounted_rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        loss = compute_loss(logits, actions, discounted_rewards)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main():
    tls_model = create_tls_model()
    memory = Memory()

    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # smoothed_reward = mdl.util.LossHistory(smoothing_factor=0.9)
    # plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Rewards')
    # TODO: plot loss. probably with Tensorboard
    #
    # if hasattr(tqdm, '_instances'):
    #     tqdm._instances.clear()  # clear if it exists
    for i_episode in tqdm(range(500)):
        # plotter.plot(smoothed_reward.get())

        # Restart the environment
        # observation = env.reset()
        # TODO: reload sumo world
        observation = numpy.array([[0, 0], [0, 0]])

        memory.clear()

        while True:
            # using our observation, choose an action and take it in the environment
            action = choose_action(tls_model, observation, single=False)
            next_observation, reward, done = numpy.array([[0, 0], [0, 0]]), numpy.array([1.0, 1.0]), True  # env.step(action)
            '''
            done - traci.simulation.getMinExpectedNumber() <= 0
            reward - sth depended on num cars on lane? something better? use vehicle value retrieval 
                   - similar to overall objective
            next_observation - simulate next 11 steps and return given world description
            '''

            '''
            maybe for reward ???
            getAccumulatedWaitingTime(self, vehID)
            getAccumulatedWaitingTime() -> double
                The accumulated waiting time of a vehicle collects the vehicle's waiting time
                over a certain time interval (interval length is set per option '--waiting-time-memory')    
            '''
            # add to memory
            memory.add_to_memory(observation, action, reward)

            # is the episode over? did you crash or do so well that you're done?
            if done:
                # smoothed_reward.append(sum(memory.rewards))  # TODO: append for plotting

                # initiate training - remember we don't know anything about how the
                #   agent is doing until it has crashed!
                train_step(tls_model, optimizer,
                           observations=np.vstack(memory.observations),
                           actions=memory.actions,
                           discounted_rewards=discount_rewards(memory.rewards))

                memory.clear()
                break

            observation = next_observation

    # saved_cartpole = mdl.lab3.save_video_of_model(cartpole_model, "CartPole-v0")
    # mdl.lab3.play_video(saved_cartpole)
    # TODO: final plotting


if __name__ == '__main__':
    main()
