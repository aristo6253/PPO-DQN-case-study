# PPO and DQN: A case study
> All the dependencies for both PPO and DQN are in the requirements.txt file. <br> *Note: some issues might arise when running the PPO trainings with `Box2D` environments, we recommend the use of Google Colab in this scenario.*

## Proximal Policy Optimization 
Implementation: Aristotelis Dimitriou

To use our implementation of the PPO algorithm in the most basic way only a few lines of code are required, here is an example:

```python
import gym

env = gym.make('Pendulum-v1')
ppo = PPO_ours(env)
ppo.learn(1_000_000)

```
> **Make sure to specify `ipynb=True` to the constructor of the class if you are running this code on a jupyter notebook!**

In `main_ppo.ipynb` you can find the full pipeline employed to generate the data using our PPO algorithm as well as the data for the random policies and the Stable Baselines 3 PPO algorithm, for 3 seeds.

Note that the final cells will not work if the necessary files are not present. Comment out the plotting functions for the files that do not exist.

## Deep Q-Network
Implementation: Alexandre Ben Ahmed Kontouli

To use our implementation of DQN as it is, please use the command:
```python
python main_dqn.py
```
This will launch the train, will output a graph of episodic rewards and when closed will launch the test on the training environment. This script will also generate a csv with the episodic reward.

To change the environment, change the name of `env` in the `main()` function of the aforementioned script. 

## Contact us
For any further questions, please contact us at `alexandre.benahmedkontouli@epfl.ch` and/or `aristotelis.dimitriou@epfl.ch`