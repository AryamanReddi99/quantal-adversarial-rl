from mushroom_rl_extensions.core.multi_agent_core import MultiAgentCore


class MASCore(MultiAgentCore):
    """
    Core which allows an MAS adversary (idx_agent=1) to create a disturbance against the protagonist (idx_agent=0)
    """

    def _step(self, render):
        action = list()

        # Protagonist action
        prot_distribution = self.agent[0].policy.distribution(self._state)
        prot_action = self.agent[0].draw_action(self._state)
        action.append(prot_action)

        # MAS Action
        if type(self.agent[1]).__name__ == "MAS":
            self.agent[1].set_prot_policy_distribution(prot_distribution)
            self.agent[1].set_prot_action(prot_action)
            adv_action = self.agent[1].draw_action(self._state)
            prot_action += adv_action
            null_action = self.mdp.info.action_space[1].high * 0
            action.append(null_action)
        else:
            adv_action = self.agent[1].draw_action(self._state)
            action.append(adv_action)

        next_state, reward, absorbing, _ = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not (self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return state, action, reward, next_state, absorbing, last
