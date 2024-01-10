import sys
import json
import struct
import socket
from poker_agent import PokerAgent
import matplotlib.pyplot as plt

recent_wins = []
average_wins = []

server_ip = "127.0.0.1"  # 德州扑克平台地址
server_port = 2333  # 德州扑克平台开放端口
room_number = int(sys.argv[1])  # 一局游戏人数
name = sys.argv[2]  # 当前程序的 AI 名字
game_number = int(sys.argv[3])  # 最大对局数量

mode = sys.argv[4]  # 模式，'train' 或 'pk'


def sendJson(request, jsonData):
    data = json.dumps(jsonData).encode()
    request.send(struct.pack("i", len(data)))
    request.sendall(data)


def recvJson(request):
    data = request.recv(4)
    length = struct.unpack("i", data)[0]
    data = request.recv(length).decode()
    while len(data) != length:
        data = data + request.recv(length - len(data)).decode()
    data = json.loads(data)
    return data


if __name__ == "__main__":
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    message = dict(
        info="connect", name=name, room_number=room_number, game_number=game_number
    )
    sendJson(client, message)
    agent = PokerAgent(mode)
    position = None
    while True:
        data = recvJson(client)
        if data["info"] == "state":
            if data["position"] == data["action_position"]:  # 轮到自己行动
                position = data["position"]
                agent.inform(data)  # 告知Agent信息
                action = agent.act(data)  # 让Agent行动
                sendJson(client, {"action": action, "info": "action"})
        elif data["info"] == "result":
            if position != None:
                agent.inform(data)  # 告知Agent信息
                print(
                    "win money: {},\tyour card: {},\topp card: {},\t\tpublic card: {}".format(
                        data["players"][position]["win_money"],
                        data["player_card"][position],
                        data["player_card"][1 - position],
                        data["public_card"],
                    )
                )
                # 记录过去1000局的平均赢钱数，并绘制为图表
                win_money = data["players"][position]["win_money"]
                recent_wins.append(win_money)
                if len(recent_wins) == 1000:
                    average_win = sum(recent_wins) / 1000
                    average_wins.append(average_win)
                    recent_wins = []
                    plt.plot(average_wins)
                    plt.title("Average Win Money")
                    plt.savefig("average_win_money.png")
                # sendJson(client, {'info': 'ready', 'status': 'start'})
        else:
            print(data)
            break
    client.close()
