import pygame
from constants import GRAVITY, FLOOR_Y, DINO_X, VELOCITY

class Dino(pygame.sprite.Sprite):
    def __init__(self, starting_y, img_path):
        pygame.sprite.Sprite.__init__(self)

        self.velocity = 0
        self.grounded = True

        img = pygame.image.load(img_path).convert_alpha()
        self.image = pygame.transform.scale_by(img, 1.5)
        self.rect = self.image.get_rect()
        self.rect.midbottom = [DINO_X, starting_y]

    def distance_to_cactus(self, cactus_group):
        if cactus_group:
            cactus = cactus_group.sprites()[0]
            distance = cactus.rect.left - self.rect.right
            if distance < 0 and len(cactus_group.sprites()) == 2:
                cactus = cactus_group.sprites()[1]
                distance = cactus.rect.left - self.rect.right

            return distance

    def jump(self):
        if self.grounded == True:
            self.grounded = False
            self.velocity = -VELOCITY

    def update(self, dt):
        # keys = pygame.key.get_pressed()
        
        # if self.grounded:
        #     if keys[pygame.K_SPACE]:

        self.rect.y += self.velocity * dt
        if self.velocity > 0:   
            self.velocity += GRAVITY*1.5 * dt
        else:
            self.velocity += GRAVITY * dt

        # Ground detection
        if self.rect.bottom >= FLOOR_Y:
            self.velocity = 0
            self.grounded = True
            self.rect.bottom = FLOOR_Y

    def position(self):
        return (DINO_X, self.rect.bottom)
    
    
    def __str__(self):
        return f"Y: {self.rect.bottom}\nGrounded: {self.grounded}\n"
