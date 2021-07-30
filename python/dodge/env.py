from random import *
import math
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, n_actions):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 4, 4), 4, 2), 2, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 4, 4), 4, 2), 2, 1)
        linear_input_size = convw * convh * 64
        self.head1 = nn.Linear(linear_input_size, 512)
        self.head2 = nn.Linear(512, n_actions)


    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head2(self.head1(x.view(x.size(0), -1)))


class Node:
    def __init__(self, pos, speed, size, color):
        self.pos = pos
        self.speed = speed
        self.dir = [0, 0]
        self.dir_sustain = 0
        self.size = size
        self.color = color


    def get_distance(self, another_node):
        return math.sqrt((self.pos[0] - another_node.pos[0]) ** 2 + (self.pos[1] - another_node.pos[1]) ** 2)


    def get_distance_by_pos(self, x, y):
        return math.sqrt((self.pos[0] - x) ** 2 + (self.pos[1] - y) ** 2)


    def is_collide(self, another_node):
        distance = self.get_distance(another_node)
        if distance <= self.size * 2:
            return True
        return False


class Environment:
    def __init__(self):
        self.window_width = 600
        self.window_height = 600

        self.initial_pos = [int(self.window_width / 2), int(self.window_height / 2)]
        self.node_size = 5
        self.player_speed = 3
        self.player = Node(self.initial_pos, self.player_speed, self.node_size, RED)

        self.num_enemy = 250
        self.enemy_list = []
        self.enemy_speed = 6
        self.enemy_pos = [] # 중복 체크를 위한 list

        # initialize enemy
        for i in range(self.num_enemy):
            while True:
                x = randint(0, self.window_width)
                y = randint(0, self.window_height)

                if not [x,y] in self.enemy_pos and self.player.get_distance_by_pos(x, y) > 20:
                    self.enemy_list.append(Node([x,y], self.enemy_speed, self.node_size, BLACK))
                    self.enemy_pos.append([x, y])
                    break


    def draw_node(self, node):
        left_top = (node.pos[0] - node.size, node.pos[1] - node.size)
        right_bottom = (node.pos[0] + node.size, node.pos[1] + node.size)
        self.draw.ellipse([left_top, right_bottom], fill=node.color)


    def move_enemy(self):
        # enemy moving
        for enemy in self.enemy_list:
            if enemy.dir_sustain == 0:
                enemy.dir_sustain = randint(1, 30)
                
                degree = randint(0, 359)
                dx = enemy.speed * math.cos(math.radians(degree))
                dy = enemy.speed * math.sin(math.radians(degree))
                
                enemy.dir[0] = int(dx)
                enemy.dir[1] = int(dy)

            enemy.pos[0] += enemy.dir[0]
            enemy.pos[1] += enemy.dir[1]

            x, y = enemy.pos
            if x < 0:
                enemy.pos[0] = 0
                enemy.dir_sustain = 1
            
            elif x > self.window_width:
                enemy.pos[0] = self.window_width - 1
                enemy.dir_sustain = 1
            
            if y < 0:
                enemy.pos[1] = 0
                enemy.dir_sustain = 1
            
            elif y > self.window_height:
                enemy.pos[1] = self.window_height - 1
                enemy.dir_sustain = 1

            enemy.dir_sustain -= 1


    def get_screen(self):
        img = Image.new("RGB", (self.window_height, self.window_width), color='#FFFFFF')
        self.draw = ImageDraw.Draw(img)

        # draw player
        self.draw_node(self.player)

        # draw enemy
        for enemy in self.enemy_list:
            self.draw_node(enemy)

        trans = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        
        tensor = trans(img)
        return tensor.unsqueeze(0)

    
    def step(self, action):
        if action == 0 and self.player.pos[0] > self.player.speed + self.node_size: # left
            self.player.pos[0] -= self.player.speed
        elif action == 1 and self.player.pos[0] < self.window_width - self.player.speed + self.node_size: # right
            self.player.pos[0] += self.player.speed
        elif action == 2 and self.player.pos[1] > self.player.speed + self.node_size: # top
            self.player.pos[1] -= self.player.speed
        elif action == 3 and self.player.pos[1] < self.window_height - self.player.speed + self.node_size: # bottom
            self.player.pos[1] += self.player.speed
        elif action == 4:
            pass

        self.move_enemy()

        collide = False
        reward = 1

        for enemy in self.enemy_list:
            if enemy.is_collide(self.player):
                collide = True
                reward = -1000000

        return reward, collide
