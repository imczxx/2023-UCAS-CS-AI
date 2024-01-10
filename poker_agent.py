# PokerAgent
# PokerAgent将json格式的data转化数字序列state，然后调用DQNAgent的act方法，得到action
# 此外，PokerAgent还记录(last_state, last_action, last_reward, current_state)四元组，用于训练DQNAgent
from dqn_agent import DQNAgent

CARD2NUM = {card: i for i, card in enumerate([rank+suit for rank in '23456789TJQKA' for suit in 'shdc'])}
ACT2NUM = {'fold': 0, 'check': 1,'call': 2, 'raise': 3}

alive_reward = 0 # 活下来的奖励

class PokerAgent:
    def __init__(self, mode='pk'):
        self.current_state = None
        # 状态：private_cards + public_cards + [my_bet, max_bet] + legal_actions
        self.state_size = 54+54+2+4
        self.action_size = 4
        self.last_state = [0]*self.state_size
        self.last_action = [0]*(self.action_size+1) # +1指的是还要记录的raise数额
        self.last_reward = [0]
        self.position = None
        self.mode = mode
        self.model = DQNAgent(self.state_size, self.action_size, mode)

    def act(self, data):
        action_mask = [0]*self.action_size
        # 生成合法动作的掩码
        for action in data['legal_actions']:
            action_mask[ACT2NUM[action]] = 1
        action = self.model.act(self.current_state, action_mask)
        #  将数字表示的动作转化为字符串表示的动作
        action = list(ACT2NUM.keys())[action]
        raise_num = 0
        if data['raise_range']:
            min_raise, max_raise = data['raise_range']
            # raise数额固定为100（可以取到的话）
            if min_raise <= 100 <= max_raise:
                raise_num = 100
            else:
                if 100 < min_raise:
                    raise_num = min_raise
                else:
                    raise_num = max_raise
        self.last_action = [0]*(self.action_size+1)
        self.last_action[ACT2NUM[action]] = 1
        self.last_action[4] = raise_num if action == 'raise' else 0
        if(action == 'raise'):
            action = 'r'+str(raise_num)
        return action

    def inform(self, data):
        if data['info'] == 'state':
            self.position = data['position']
            self.last_reward = [alive_reward]
            private_cards = [0] * 54
            for card in data['private_card']:
                private_cards[CARD2NUM[card]] = 1

            public_cards = [0] * 54
            for card in data['public_card']:
                public_cards[CARD2NUM[card]] = 1

            legal_actions = [0] * 4
            for action in data['legal_actions']:
                legal_actions[ACT2NUM[action]] = 1
            # my_bet：我已经下注的数额，max_bet：其他人已经下注的最大数额
            my_bet = [player['total_money'] - player['money_left'] for player in data['players'] if player['position'] == data['position']][0]
            max_bet = max([player['total_money'] - player['money_left'] for player in data['players'] if player['position'] != data['position']])

            self.current_state = private_cards + public_cards + [my_bet, max_bet] + legal_actions
            transition = (self.last_state, self.last_action, self.last_reward, self.current_state)
            if(self.last_action != [0]*(self.action_size+1) and  self.mode == 'train'):
                self.model.feed(transition)
            self.last_state = self.current_state

        elif data['info'] == 'result':
            self.last_reward = [data['players'][self.position]['win_money']]
            self.current_state = [0]*self.state_size
            transition = (self.last_state, self.last_action, self.last_reward, self.current_state)
            if(self.last_action != [0]*(self.action_size+1) and  self.mode == 'train'):
                self.model.feed(transition)
            self.last_state = [0]*self.state_size
            self.last_action = [0]*(self.action_size+1)