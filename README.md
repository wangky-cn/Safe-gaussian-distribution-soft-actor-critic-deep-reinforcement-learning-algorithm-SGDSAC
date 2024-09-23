GDSAC is an algorithm based on the soft actor-critic (SAC) framework, with its novelty lying in the learning of the Q-value distribution (mean and variance).

1. GDSAC provides a formula for estimating the target variance of Q-values under the maximum entropy reinforcement learning framework, enabling GDSAC to use various distance formulas to construct the loss function for the Q-network.
2. GDSAC simultaneously learns two Q-value distributions and reduces the estimation bias of Q-values and variance through minimum variance weighting.
3. To avoid unreasonable updates to the Q-values, GDSAC uses the 3-sigma rule to limit the magnitude of Q-value updates.
4. GDSAC utilizes upper confidence bounds to encourage exploration.

The authors have conducted preliminary tests of GDSAC's performance in different environments of a certain control problem and compared it with SAC and TD3, demonstrating GDSAC's strong performance. According to current experimental results, GDSAC shows significant advantages in learning speed and stability.

In the coming months, the authors plan to open-source the final version of GDSAC and present detailed experimental results.

If you have any suggestions, the authors would appreciate hearing from you.
E-mail: wangky27@163.com
