import pygame
from image_utils import load_image

class Cactus(pygame.sprite.Sprite):

    def __init__(self, x, y, img_path):
        pygame.sprite.Sprite.__init__(self)
        img = load_image(img_path)
        self.image = pygame.transform.scale_by(img, 1.5)
        self.rect = self.image.get_rect()
        self.rect.midbottom = [x, y]
        self.passed = False

    def update(self, game_speed, dt, cactus_passed, dino_left):
        self.rect.x -= game_speed * dt
        if not self.passed and self.rect.right < dino_left:
            cactus_passed["count"] += 1
            self.passed = True
        if self.rect.right < 0:
            self.kill()