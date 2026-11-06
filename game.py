import pygame
from constants import WINDOW_HEIGHT, WINDOW_WIDTH, FLOOR_Y, FPS, GAME_SPEED
from dino import Dino
from environment import Environment
import random

class Game:
    def __init__(self, render_mode = True):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.dino = Dino(FLOOR_Y, "assets/dino.png")
        self.environment = Environment(self.screen, self.dino, GAME_SPEED)
        print(self.environment.get_state())
        self.fps = 60
        self.render_mode = render_mode
        self.clock.tick(FPS)

        self.total_jumps = 0
        self.reset()

    def reset(self):
        # self.environment.reset()
        self.dino = Dino(FLOOR_Y, "assets/dino.png")
        self.environment = Environment(self.screen, self.dino, GAME_SPEED)
        self.dt = 1.0 / self.fps

        return self.environment.get_state(), {}

    def get_random_action(self):
        return random.choice([0,1])


    def step(self,action, reward):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                pygame.quit()
                exit()

        if action == 1:
            self.environment.dino.jump()

        self.environment.add_cactus()
        self.environment.update(self.dt)    
        self.environment.draw(action, reward)

        

        terminated = self.environment.is_collision()
        truncated = False

        if terminated:
            reward = -100.0
        else:

            distance = self.environment.dino.distance_to_cactus(self.environment.cactus_group)
            # reward = 0.1 * 1/self.dino.jump_count
            if self.environment.dino.grounded:
                reward = 0.3
            else:
                reward =0.01

            if action == 1:
                if distance is None or distance > 300:
                    print("penalty")
                    reward += -10.0 

        if self.render_mode:
            self.environment.screen.fill((255, 255, 255)) # Clear screen
            self.environment.draw(action, reward)
            pygame.display.flip()
            self.dt = self.clock.tick(self.fps) / 1000.0
        
        new_state = self.environment.get_state()
        info = {}

        return new_state, reward, terminated, truncated, info