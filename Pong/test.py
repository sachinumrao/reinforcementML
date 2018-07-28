import pygame
from pygame.locals import *

def main():
    pygame.init()

    screensize = (640,480)
    screen = pygame.display.set_mode(screensize)
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == QUIT:
                print("Quitting the app\n")
                pygame.quit()
                running = False

            if event.type == pygame.K_UP:
                print("Button Pressed")

            if event.type == pygame.K_DOWN:
                print("Button Released")

main()


