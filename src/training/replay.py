import random
import numpy as np


class ReplayBuffer:
    """
    ## Buffer for Prioritized Experience Replay

    [Prioritized experience replay](https://papers.labml.ai/paper/1511.05952)
     samples important transitions more frequently.
    The transitions are prioritized by the Temporal Difference error (td error), $\delta$.

    """

    def __init__(self, capacity, alpha, obs_length):
        """
        ### Initialize
        """

        self.capacity = capacity

        self.alpha = alpha
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.

        self.data = {
            'obs': np.zeros(shape=(capacity, obs_length), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'reward': np.zeros(shape=capacity, dtype=np.float32),
            'next_obs': np.zeros(shape=(capacity, obs_length), dtype=np.uint8),
            'done': np.zeros(shape=capacity, dtype=np.bool_)
        }

        self.next_idx = 0
        self.size = 0


    def add(self, obs, action, reward, next_obs, done):
        """
        ### Add sample to queue
        """

        idx = self.next_idx

        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        priority_alpha = self.max_priority ** self.alpha

        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)


    def _set_priority_min(self, idx, priority_alpha):
        """
        #### Set priority in binary segment tree for minimum
        """

        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])


    def _set_priority_sum(self, idx, priority):
        """
        #### Set priority in binary segment tree for sum
        """

        idx += self.capacity
        self.priority_sum[idx] = priority

        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]


    def _sum(self):
        """
        #### $\sum_k p_k^\alpha$
        """

        return self.priority_sum[1]
    

    def _min(self):
        """
        #### $\min_k p_k^\alpha$
        """

        return self.priority_min[1]
    

    def find_prefix_sum_idx(self, prefix_sum):
        """
        #### Find largest $i$ such that $\sum_{k=1}^{i} p_k^\alpha  \le P$
        """

        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity

    def sample(self, batch_size, beta):
        """
        ### Sample from buffer
        """

        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indices': np.zeros(shape=batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indices'][i] = idx

        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indices'][i]

            prob = self.priority_sum[idx + self.capacity] / self._sum()

            weight = (prob * self.size) ** (-beta)

            samples['weights'][i] = weight / max_weight


        for k, v in self.data.items():
            samples[k] = v[samples['indices']]

        return samples


    def update_priorities(self, indexes, priorities):
        """
        ### Update priorities
        """

        for idx, priority in zip(indexes, priorities):

            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority ** self.alpha

            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)
