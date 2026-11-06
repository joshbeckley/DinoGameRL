import pygame
from constants import FLOOR_Y, CACTUS_FREQUENCY_SECS, BLACK, WHITE, GAME_SPEED_INCREASE, GAME_SPEED
import random
from cactus import Cactus
from ground import Ground
from visualizer import NeuralNetworkVisualizer

class Environment:

    def __init__(self, screen, dino, game_speed):
        self.screen = screen
        self.dino = dino
        self.game_speed = game_speed
        self.score = 0

        self.font = pygame.font.Font('freesansbold.ttf', 10)
        self.visualizer = NeuralNetworkVisualizer()
        self.dino_group = pygame.sprite.Group()
        self.dino_group.add(self.dino)

        self.cactus_group = pygame.sprite.Group()

        self.ground = Ground(0, FLOOR_Y, "assets/ground.png")
        # TODO: could move in draw
        self.ground_group = pygame.sprite.Group()
        self.ground_group.add(self.ground)

        # self.last_cactus = pygame.time.get_ticks() - CACTUS_FREQUENCY
        self.elapsed_game_time = 0
        self.spawn_time_interval = CACTUS_FREQUENCY_SECS
        self.spawn_time_interval_decrease = 0.03
        self.next_spawn_time = 0

        self.cactus_passed = {"count": 0}

    # def reset(self):
    #     self.dino.grounded = True
    #     self.game_speed = GAME_SPEED
    #     self.score = 0
    #     self.dino_group = pygame.sprite.Group()
    #     self.dino_group.add(self.dino)

    #     self.cactus_group = pygame.sprite.Group()

    #     self.ground = Ground(0, FLOOR_Y, "assets/ground.png")
    #     # TODO: could move in draw
    #     self.ground_group = pygame.sprite.Group()
    #     self.ground_group.add(self.ground)

    #     # self.last_cactus = pygame.time.get_ticks() - CACTUS_FREQUENCY
    #     self.elapsed_game_time = 0
    #     self.spawn_time_interval = CACTUS_FREQUENCY_SECS
    #     self.spawn_time_interval_decrease = 0.03
    #     self.next_spawn_time = 0
    
    def get_state(self):
        grounded_num = int(self.dino.grounded)
        distance = 0 if self.dino.distance_to_cactus(self.cactus_group) is None else self.dino.distance_to_cactus(self.cactus_group)
        return [grounded_num, self.game_speed - GAME_SPEED , distance , FLOOR_Y - self.dino.rect.midbottom[1], self.cactus_passed["count"]]
        
    def add_cactus(self):
        # print(f"Elapsed: {self.elapsed_game_time}\nNext: {self.next_spawn_time}\n")
        if self.elapsed_game_time > self.next_spawn_time:
            random_x = random.randint(30,200)
            cactus = Cactus(self.screen.get_width() + 2 + random_x, FLOOR_Y, "assets/cactus.png")
            self.cactus_group.add(cactus)

            self.spawn_time_interval -= self.spawn_time_interval_decrease;
            self.next_spawn_time += self.spawn_time_interval;

    def is_collision(self):
        if pygame.sprite.groupcollide(self.dino_group, self.cactus_group, False, False):
            return True

        return False

    def draw(self, action, reward):
        self.screen.fill("white")

        self.visualizer.draw(self.screen,self.get_state(), action)
        self.draw_info(reward)
        self.dino_group.draw(self.screen)
        self.cactus_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        pygame.display.update()

    def update(self, dt):
        self.dino_group.update(dt)
        self.cactus_group.update(self.game_speed, dt, self.cactus_passed)
        self.game_speed += GAME_SPEED_INCREASE * dt

        self.elapsed_game_time += 1 * dt
        self.score = self.elapsed_game_time * 10

    def draw_info(self, reward):
        reward = self.font.render("reward:".ljust(30) + f"{reward}", True, BLACK, WHITE)
        rewardRect = reward.get_rect()
        rewardRect.topleft = (10, 10)
        self.screen.blit(reward, rewardRect)

        # spawn = self.font.render(f"Spawn Interval:".ljust(24) + f"{self.spawn_time_interval}", True, BLACK, WHITE)
        # spawnRect = spawn.get_rect()
        # spawnRect.topleft = (10, 20)
        # self.screen.blit(spawn, spawnRect)

        # next = self.font.render(f"Next Spawn:".ljust(26) + f"{self.next_spawn_time}", True, BLACK, WHITE)
        # nextRect = next.get_rect()
        # nextRect.topleft = (10, 30)
        # self.screen.blit(next, nextRect)

        # elapsed = self.font.render(f"Elapsed Time:".ljust(25) + f"{self.elapsed_game_time}", True, BLACK, WHITE)
        # elapsedRect = elapsed.get_rect()
        # elapsedRect.topleft = (10, 40)
        # self.screen.blit(elapsed, elapsedRect)

        # score = self.font.render(f"Score:".ljust(31) + f"{self.score}", True, BLACK, WHITE)
        # scoreRect = score.get_rect()
        # scoreRect.topleft = (10, 50)
        # self.screen.blit(score, scoreRect)

        # distance = self.font.render(f"Distance:".ljust(29) + f"{self.dino.distance_to_cactus(self.cactus_group)}", True, BLACK, WHITE)
        # distanceRect = distance.get_rect()
        # distanceRect.topleft = (10, 60)
        # self.screen.blit(distance, distanceRect)
        