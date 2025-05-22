from __future__ import annotations
import math
import time
import random
from copy import deepcopy
from agent import Agent
from battle import BattleState
from card import Card
from action.action import EndAgentTurn, PlayCard
from game import GameState
from ggpa.ggpa import GGPA
from config import Verbose


# You only need to modify the TreeNode!
class TreeNode:
    # You can change this to include other attributes. 
    # param is the value passed via the -p command line option (default: 0.5)
    # You can use this for e.g. the "c" value in the UCB-1 formula
    def __init__(self, param, parent=None):
        self.children: list[tuple[PlayCard | EndAgentTurn, TreeNode]] = []
        self.parent = parent
        self.results: list[float] = []
        self.visits = 0
        self.param = param
    
    # REQUIRED function
    # Called once per iteration
    def step(self, state):
        self.select(state)
        
    # REQUIRED function
    # Called after all iterations are done; should return the 
    # best action from among state.get_actions()
    def get_best(self, state):
        if not self.children:
            return random.choice(state.get_actions())
        best_action, best_node = max(
            self.children,
            key=lambda ac: (sum(ac[1].results) / ac[1].visits) if ac[1].visits > 0 else -math.inf
        )
        return best_action
        
    # REQUIRED function (implementation optional, but *very* helpful for debugging)
    # Called after all iterations when the -v command line parameter is present
    def print_tree(self, indent=0):
        pad = " " * indent
        for action, node in self.children:
            avg = sum(node.results) / node.visits if node.visits > 0 else 0.0
            print(f"{pad}{action}: visits={node.visits}, avg={avg:.2f}")
            node.print_tree(indent + 2)

    # RECOMMENDED: select gets all actions available in the state it is passed
    # If there are any child nodes missing (i.e. there are actions that have not 
    # been explored yet), call expand with the available options
    # Otherwise, pick a child node according to your selection criterion (e.g. UCB-1)
    # apply its action to the state and recursively call select on that child node.
    def select(self, state):
        if state.ended():
            reward = self.score(state)
            self.backpropagate(reward)
            return self

        actions = state.get_actions()

        if len(self.children) < len(actions):
            return self.expand(state, actions)

        total_visits = sum(node.visits for (_, node) in self.children)
        log_total = math.log(total_visits) if total_visits > 0 else 0.0

        best_score = -math.inf
        best_action = None
        best_node = None

        for action, node in self.children:
            if action not in actions:
                continue
            if node.visits == 0:
                score = math.inf
            else:
                avg_reward = sum(node.results) / node.visits
                exploit = avg_reward
                explore = self.param * math.sqrt(log_total / node.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_action = action
                best_node = node

        if best_action is None:
            print(f"[DEBUG] no UCB pick; legal actions = {actions}")
            print(f"[DEBUG] children actions    = {[a for a, _ in self.children]}")
            best_action = random.choice(actions)
            for a, n in self.children:
                if a == best_action:
                    best_node = n
                    break
            else:
                best_node = TreeNode(self.param, parent=self)
                self.children.append((best_action, best_node))

        state.step(best_action)
        return best_node.select(state)

    # RECOMMENDED: expand takes the available actions, and picks one at random,
    # adds a child node corresponding to that action, applies the action to the state
    # and then calls rollout on that new node
    def expand(self, state, available):
        tried = [act for act, _ in self.children]
        unexplored = [a for a in available if a not in tried]
        action = random.choice(unexplored)

        child = TreeNode(self.param, parent=self)
        self.children.append((action, child))

        state.step(action)
        reward = child.rollout(state)
        child.backpropagate(reward)
        return child

    # RECOMMENDED: rollout plays the game randomly until its conclusion, and then 
    # returns the result
    def rollout(self, state):
        while not state.ended():
            action = random.choice(state.get_actions())
            state.step(action)
        return self.score(state)
        
    # RECOMMENDED: backpropagate records the score you got in the current node, and 
    # then recursively calls the parent's backpropagate as well.
    # If you record scores in a list, you can use sum(self.results)/len(self.results)
    # to get an average.
    def backpropagate(self, result):
        self.results.append(result)
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        
    # RECOMMENDED: default scoring function
    # optimizing; for the challenge scenario, in particular, you may want to experiment
    # with other options (e.g. squaring the score, or incorporating state.health(), etc.)
    def score(self, state): 
        return state.score()
        

# You do not have to modify the MCTS Agent (but you can)
class MCTSAgent(GGPA):
    def __init__(self, iterations: int, verbose: bool, param: float):
        self.iterations = iterations
        self.verbose = verbose
        self.param = param

    # REQUIRED METHOD
    def choose_card(self, game_state: GameState, battle_state: BattleState) -> PlayCard | EndAgentTurn:
        actions = battle_state.get_actions()
        if len(actions) == 1:
            return actions[0].to_action(battle_state)
    
        t = TreeNode(self.param)
        start_time = time.time()

        for _ in range(self.iterations):
            sample_state = battle_state.copy_undeterministic()
            t.step(sample_state)
        
        best_action = t.get_best(battle_state)
        if self.verbose:
            t.print_tree()
        
        if best_action is None:
            print("WARNING: MCTS did not return any action")
            return random.choice(self.get_choose_card_options(game_state, battle_state))
        return best_action.to_action(battle_state)
    
    # REQUIRED METHOD: All our scenarios only have one enemy
    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        return agent_list[0]
    
    # REQUIRED METHOD: Our scenarios do not involve targeting cards
    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        return card_list[0]
