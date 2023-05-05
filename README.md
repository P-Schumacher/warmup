# warmup
Gym environments for musculoskeletal reaching tasks.
More detailed README will follow.

## Environments
Available environments are:
`muscle_arm-v0` `torque_arm-v0`, `humanreacher-v0`

## Example code

```
import gym
import warmup

env = gym.make("humanreacher-v0")

for ep in range(5):
     ep_steps = 0
     state = env.reset()
     while True:
         next_state, reward, done, info = env.step(env.action_space.sample())
         env.render()
         if done or (ep_steps >= env.max_episode_steps):
             break
         ep_steps += 1
```
