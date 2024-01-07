import sys
import json
import struct
import socket
import random
import cProfile

server_ip = "127.0.0.1"                 # 德州扑克平台地址
server_port = 2333                      # 德州扑克平台开放端口
room_number = int(sys.argv[1])          # 一局游戏人数
name = sys.argv[2]                      # 当前程序的 AI 名字
game_number = int(sys.argv[3])          # 最大对局数量


def get_action(data):
    # print(data)
    legal_actions = data['legal_actions']
    action = random.choice(legal_actions)
    if(action == 'raise'):
        if(random.random() < 0.1):
            action = 'r' + str(data['raise_range'][1])
        else:
            action = 'r' + str(random.randint(data['raise_range'][0], int((data['raise_range'][0]+data['raise_range'][1])/2)))
    return action


def sendJson(request, jsonData):
    data = json.dumps(jsonData).encode()
    request.send(struct.pack('i', len(data)))
    request.sendall(data)


def recvJson(request):
    data = request.recv(4)
    length = struct.unpack('i', data)[0]
    data = request.recv(length).decode()
    while len(data) != length:
        data = data + request.recv(length - len(data)).decode()
    data = json.loads(data)
    return data


def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    message = dict(info='connect',
                   name=name,
                   room_number=room_number,
                   game_number=game_number)
    sendJson(client, message)
    position = 1
    while True:
        data = recvJson(client)
        if data['info'] == 'state':
            if data['position'] == data['action_position']:
                position = data['position']
                action = get_action(data)
                sendJson(client, {'action': action, 'info': 'action'})
        elif data['info'] == 'result':
            print('win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}'.format(
                data['players'][position]['win_money'], data['player_card'][position],
                data['player_card'][1 - position], data['public_card']))
            # sendJson(client, {'info': 'ready', 'status': 'start'})
        else:
            print(data)
            break
    client.close()

cProfile.run('main()')
