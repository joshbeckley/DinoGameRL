import pygame
from image_utils import load_image

class Ground(pygame.sprite.Sprite):

    def __init__(self, x, y, img_path):
        pygame.sprite.Sprite.__init__(self)
        img = load_image(img_path)
        self.image = pygame.transform.scale_by(img, 1.5)
        self.rect = self.image.get_rect()
        self.rect.topleft = [x, y-8]