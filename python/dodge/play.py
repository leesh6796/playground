from random import *
import math
from sim import Simulator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from env import *
from sim import *
import sys


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

DISPLAY = False


class RLPlayer:
    def __init__(self, h, w, n_actions):
        self.n_actions = n_actions

        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.gamma = 0.75
        self.lr = 0.002
        self.batch_size = 32
        self.target_update = 10

        self.policy_net = DQN(h, w, self.n_actions).to(device)
        self.target_net = DQN(h, w, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)


    def select_action(self, state): # state는 image tensor
        sample = random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
                # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
                # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[randrange(self.n_actions)]], device=device, dtype=torch.long)

    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
        # 전환합니다.
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def train(self, n_episodes):
        env = Environment()
        for i_episode in range(n_episodes):
            print(i_episode)
            sum_reward = 0
            env = Environment()
            last_screen = env.get_screen()
            current_screen = env.get_screen()
            state = current_screen - last_screen

            for t in count():
                action = self.select_action(state)
                reward, done = env.step(action)
                sum_reward += reward
                reward = torch.tensor([reward], device=device)

                last_screen = current_screen
                current_screen = env.get_screen()

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()
                if done:
                    print("점수: " + str(sum_reward + 1000000))
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode % 10 == 0:
                torch.save(self.policy_net.state_dict(), "policy_net.model")
                torch.save(self.target_net.state_dict(), "target_net.model")


mode = "inference"

player = RLPlayer(600, 600, 5)

if mode == "train":
    player.train(2000)

    torch.save(player.policy_net.state_dict(), "policy_net.model")
    torch.save(player.target_net.state_dict(), "target_net.model")

elif mode == "inference":
    player.policy_net.load_state_dict(torch.load("./policy_net.model"))
    player.policy_net.eval()

    player.target_net.load_state_dict(torch.load("./target_net.model"))
    player.target_net.eval()

    sim = Simulator()
    sim.initialize()
    last_screen = sim.get_screen()

    pygame.key.set_repeat(100, 0)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == KEYDOWN:
                current_screen = sim.get_screen()
                state = current_screen - last_screen

                action = player.select_action(state)[0].item()
                is_game_end = sim.select_action(action)

                if is_game_end:
                    print("충돌!")
                    pygame.quit()
                    sys.exit()