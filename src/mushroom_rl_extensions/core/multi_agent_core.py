import numpy as np
from mushroom_rl_extensions.core.core import Core


class MultiAgentCore(Core):
    def _run_impl(
        self,
        move_condition,
        fit_condition_per_agent,
        steps_progress_bar,
        episodes_progress_bar,
        render,
        initial_states,
    ):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter_per_agent = np.zeros(len(self.agent), dtype=int)
        self._current_steps_counter_per_agent = np.zeros(len(self.agent), dtype=int)

        dataset_per_agent = [list() for _ in range(len(self.agent))]
        last = True
        while move_condition():
            if last:
                self.reset(initial_states)

            sample = self._step(render)

            # Quadruped Success
            if "Quadruped" in type(self.mdp.env.task).__name__:
                goal_pos = np.array([10, 0])
                torso_pos = self.mdp.env.physics.named.data.geom_xpos["torso"][:2]
                dist = np.linalg.norm(goal_pos - torso_pos)
                success = dist < 1.0

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter_per_agent += 1
            steps_progress_bar.update(1)

            last = sample[-1]
            if last:
                self._total_episodes_counter += 1
                self._current_episodes_counter_per_agent += 1
                episodes_progress_bar.update(1)

            [
                dataset.append(sample)
                for idx_agent, dataset in enumerate(dataset_per_agent)
            ]

            for idx_agent, fit_condition in enumerate(fit_condition_per_agent):
                if fit_condition():
                    self.agent[idx_agent].fit(dataset_per_agent[idx_agent])
                    self._current_episodes_counter_per_agent[idx_agent] = 0
                    self._current_steps_counter_per_agent[idx_agent] = 0

                    if idx_agent == 0:
                        for c in self.callbacks_fit:
                            c(dataset_per_agent[0])
                    else:
                        pass
                        # ToDo: Introduce callbacks for adversary (?)
                    dataset_per_agent[
                        idx_agent
                    ] = (
                        list()
                    )  # fit stores data in agent replay buffer, so core's replay buffer can be reset

        for agent in self.agent:
            agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset_per_agent[0]  # just protagonist dataset

    def _step(self, render):
        action = list()
        for idx_agent in range(len(self.agent)):
            action.append(self.agent[idx_agent].draw_action(self._state))
            self.action_norms[idx_agent].append(np.linalg.norm(action[idx_agent]))

        next_state, reward, absorbing, info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            render_info = {"action": action, "reward": reward}
            self.mdp.render(render_info)

        last = not (self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return state, action, reward, next_state, absorbing, last

    def reset(self, initial_states=None):
        """
        Reset the state of the mdp and agents.

        """
        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        for agent in self.agent:
            agent.episode_start()
            agent.next_action = None
        self._episode_steps = 0
