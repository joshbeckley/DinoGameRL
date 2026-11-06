import pygame
from constants import WINDOW_HEIGHT, WINDOW_WIDTH, FLOOR_Y, FPS, GAME_SPEED
from dino import Dino
from environment import Environment
import random

class Game:
    def __init__(self, render_mode = True):
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.dino = Dino(FLOOR_Y, "assets/dino.png")
        self.environment = Environment(screen, self.dino, GAME_SPEED)
        print(self.environment.get_state())
        self.fps = 60
        self.render_mode = render_mode
        self.clock.tick(FPS)

        self.reset()

    def reset(self):
        self.dino.grounded = True
        self.environment.game_speed = GAME_SPEED
        self.environment.score = 0
        self.dt = 1.0 / self.fps

        return self.environment.get_state(), {}

    def get_random_action(self):
        return random.choice([0,1])


    def step(self,action):

        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if action == 1:
            self.environment.dino.jump()

        self.environment.add_cactus()
        self.environment.update(self.dt)    
        self.environment.draw()

        

        terminated = self.environment.is_collision()
        truncated = False

        if terminated:
            reward = -100.0
        else:
            reward = 0.1

        if self.render_mode:
            self.environment.screen.fill((255, 255, 255)) # Clear screen
            self.environment.draw()
            pygame.display.flip()
            self.dt = self.clock.tick(self.fps) / 1000.0
        
        new_state = self.environment.get_state()
        info = {}

        return new_state, reward, terminated, truncated, info