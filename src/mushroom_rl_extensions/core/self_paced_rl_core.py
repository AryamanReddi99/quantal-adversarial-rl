from mushroom_rl_extensions.core.multi_agent_core import MultiAgentCore


class SelfPacedRLCore(MultiAgentCore):
    def set_variables_for_training(self, new_variables):
        for preprocessor in self._preprocessors:
            preprocessor.reset_episode_count()
            preprocessor.set_variables(new_variables)

    def reset(self, initial_states=None):
        """
        Update the state preprocessors.
        Reset the state of the mdp and agents.

        """
        for preprocessor in self._preprocessors:
            preprocessor.update_episode_count(self._total_episodes_counter)

        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        for agent in self.agent:
            agent.episode_start()
            agent.next_action = None
        self._episode_steps = 0
