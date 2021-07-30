import pygame
from pygame.locals import *
from random import *
import math
import torch
from torchvision import transforms

from PIL import Image, ImageDraw


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)


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


class Simulator:
    def __init__(self):
        self.window_width = 600
        self.window_height = 600

        self.FPS = 30
        self.clock = pygame.time.Clock()

        self.initial_pos = [int(self.window_width / 2), int(self.window_height / 2)]
        self.node_size = 5
        self.player_speed = 3
        self.player = Node(self.initial_pos, self.player_speed, self.node_size, RED)

        self.num_enemy = 200
        self.enemy_list = []
        self.enemy_speed = 3
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
        pygame.draw.circle(self.screen, node.color, node.pos, self.node_size)


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

    
    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), DOUBLEBUF)

        self.screen.fill(WHITE)

        # draw player
        self.draw_node(self.player)

        # draw enemy
        for enemy in self.enemy_list:
            self.draw_node(enemy)

        pygame.display.flip()
        self.clock.tick(self.FPS)

    
    def get_screen(self):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        bytes = pygame.image.tostring(self.screen, "RGB", False)
        img = Image.frombytes("RGB", self.screen.get_size(), bytes)
        tensor = trans(img)
        return tensor.unsqueeze(0)

    
    def select_action(self, action):
        if action == 0 and self.player.pos[0] > self.player.speed + self.node_size: # left
            self.player.pos[0] -= self.player.speed
        elif action == 1 and self.player.pos[0] < self.window_width - self.player.speed + self.node_size: # right
            self.player.pos[0] += self.player.speed
        elif action == 2 and self.player.pos[1] > self.player.speed + self.node_size: # top
            self.player.pos[1] -= self.player.speed
        elif action == 3 and self.player.pos[1] < self.window_height - self.player.speed + self.node_size: # bottom
            self.player.pos[1] += self.player.speed
        elif action == 4: # stop
            pass

        self.move_enemy()
        self.screen.fill(WHITE)

        # draw player
        self.draw_node(self.player)

        # draw enemy
        for enemy in self.enemy_list:
            self.draw_node(enemy)
            if enemy.is_collide(self.player):
                return True

        pygame.display.flip()
        self.clock.tick(self.FPS)

        return False