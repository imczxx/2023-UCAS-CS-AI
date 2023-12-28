from dqn import DQN_Model
import random
card_dict = {card: i for i, card in enumerate([rank+suit for rank in '23456789TJQKA' for suit in 'shdc'])}
action_dict = {'fold': 0, 'check': 1, 'raise': 2}

class Agent:
    def __init__(self):
        self.state = None
        self.model = DQN_Model()

    def act(self, data):
        actions = DQN_Model.eval(self.state)
        
        legal_actions, raise_amount = data['legal_actions']
        actions = {k: v for k, v in actions.items() if k in legal_actions}
        total_prob = sum(actions.values())
        actions = {k: v / total_prob for k, v in actions.items()}

        action = random.choices(list(actions.keys()), list(actions.values()))[0]
        
        if(action == 'raise'):
            action = 'r+' + str(raise_amount)
        
        return action
    
    def inform(self, data):
        private_cards = list(range(54))
        for card in data['private_card']:
            private_cards[card_dict[card]] = 1

        public_cards = list(range(54))
        for card in data['public_card']:
            public_cards[card_dict[card]] = 1

        my_bet = [player['total_money'] - player['money_left'] for player in data['players'] if player['position'] == data['position']][0]
        max_bet = max([player['total_money'] - player['money_left'] for player in data['players'] if player['position'] != data['position']])

        legal_actions = list(range(3))
        for action in data['legal_actions']:
            legal_actions[action_dict[action]] = 1

        self.state = private_cards + public_cards + [my_bet, max_bet] + legal_actions