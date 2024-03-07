from gymnasium.envs.registration import register

register(
    id="WindyPointMass",
    entry_point="windy_point_mass.WindyPointMass",
    max_episode_steps=300,
)
