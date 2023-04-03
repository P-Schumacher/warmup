from gym.envs.registration import register

register(
    id="muscle_arm-v0",
    entry_point="warmup.envs:MuscleArmMuJoCo",
    max_episode_steps=300,
)


register(
    id="torque_arm-v0",
    entry_point="warmup.envs:TorqueArmMuJoCo",
    max_episode_steps=300,
)


register(
    id="humanreacher-v0",
    entry_point="warmup.envs:HumanReacher",
    max_episode_steps=300,
)
