from dqn_agent import DQNAgent

card_dict = {card: i for i, card in enumerate([rank+suit for rank in '23456789TJQKA' for suit in 'shdc'])}
action_dict = {'fold': 0, 'check': 1,'call': 2, 'raise': 3}

alive_reward = 1 # which value is better?

class PokerAgent:
    def __init__(self):
        self.current_state = None
        self.state_size = 54+54+2+4
        self.action_size = 4
        self.last_state = [0]*self.state_size
        self.last_action = [0]*(self.action_size+1)
        self.last_reward = [0]
        self.position = None
        self.model = DQNAgent(self.state_size, self.action_size)

    def act(self, data):
        action_mask = [0] * 4
        for action in data['legal_actions']:
            action_mask[action_dict[action]] = 1
        action, raise_amount = self.model.act(self.current_state, action_mask)
        action = list(action_dict.keys())[action]
        self.last_action = [0]*5
        self.last_action[action_dict[action]] = 1
        self.last_action[4] = raise_amount if action == 'raise' else 0
        if(action == 'raise'):
            action = 'r'+str(raise_amount)
        return action

    def inform(self, data):
        if data['info'] == 'state':
            self.position = data['position']
            self.last_reward = [alive_reward]
            private_cards = [0] * 54
            for card in data['private_card']:
                private_cards[card_dict[card]] = 1
            
            public_cards = [0] * 54
            for card in data['public_card']:
                public_cards[card_dict[card]] = 1

            legal_actions = [0] * 4
            for action in data['legal_actions']:
                legal_actions[action_dict[action]] = 1

            my_bet = [player['total_money'] - player['money_left'] for player in data['players'] if player['position'] == data['position']][0]
            max_bet = max([player['total_money'] - player['money_left'] for player in data['players'] if player['position'] != data['position']])
            
            self.current_state = private_cards + public_cards + [my_bet, max_bet] + legal_actions
            transition = (self.last_state, self.last_action, self.last_reward, self.current_state)
            if(self.last_action != [0]*(self.action_size+1)):
                self.model.feed(transition)
            self.last_state = self.current_state

        elif data['info'] == 'result':
            self.last_reward = [data['players'][self.position]['win_money']]
            self.current_state = [0]*self.state_size
            transition = (self.last_state, self.last_action, self.last_reward, self.current_state)
            if(self.last_action != [0]*(self.action_size+1)):
                self.model.feed(transition)
            self.last_state = [0]*self.state_size
            self.last_action = [0]*(self.action_size+1)
            
            
            