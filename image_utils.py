import pygame


def load_image(path):
    image = pygame.image.load(path)
    if pygame.display.get_surface() is not None:
        return image.convert_alpha()
    return image
