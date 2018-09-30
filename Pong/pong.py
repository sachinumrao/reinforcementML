#import dependencies for game
import numpy as np
import pygame
from pygame.locals import *


class Pong(object):

    #initializa the object, 'self' is referencing to object itself
    def __init__(self, screensize):

        self.cent_x = int(screensize[0]*0.5)
        self.cent_y = int(screensize[1]*0.5)

        #radius of the pong ball
        self.radius = 8

        self.screensize = screensize
        self.rect = pygame.Rect(self.cent_x - self.radius,
                                self.cent_y - self.radius,
                                self.radius*2, self.radius*2)

        self.color = (100,100,255)

        self.direction = [1,1]

        self.speed_x = 8
        self.speed_y = 6

        self.hit_edge_left = False
        self.hit_edge_right = False
        #Code Task: change speed as game progresses

    def update(self, player_paddle = None, ai_paddle=None):
        self.cent_x += self.direction[0]*self.speed_x
        self.cent_y += self.direction[1]*self.speed_y

        self.rect.center = (self.cent_x, self.cent_y)

        #change direction on collision with top or bottom
        if self.rect.top <=0:
            self.direction[1] = 1

        elif self.rect.bottom >= self.screensize[1]-1:
            self.direction[1] = -1

        #hit the edge and the game is over
        if self.rect.right >= self.screensize[0]-1:
            self.hit_edge_right = True
        elif self.rect.left <= 0:
            self.hit_edge_left = True


        #Check the collisions with paddle
        if self.rect.colliderect(player_paddle.rect):
            self.direction[0] = -1


        if self.rect.colliderect(ai_paddle.rect):
            self.direction[0] = 1



    def render(self, screen):
        pygame.draw.circle(screen, self.color, self.rect.center, self.radius, 0)
        pygame.draw.circle(screen, (255,255,255), self.rect.center, self.radius, 1)
        #print("Cricel Drawn...")




#AI paddle
class AIPaddle(object):

    def __init__(self, screensize):

        self.screensize = screensize

        self.center_x = 5
        self.center_y = int(screensize[1]*0.5)

        self.height = 100
        self.width = 10


        self.rect = pygame.Rect(0,self.center_y - int(self.height*0.5) , self.width, self.height)
        self.color = (100,255,100)

        self.speed = 6

    def update(self, pong):

        if pong.rect.top < self.rect.top:
            self.center_y -= self.speed

        elif pong.rect.bottom > self.rect.bottom:
            self.center_y += self.speed

        self.rect.center = (self.center_x, self.center_y)

    def render(self, screen):

        pygame.draw.rect(screen, self.color, self.rect, 0)
        pygame.draw.rect(screen, (0,0,0), self.rect, 1)
        #print("AI_paddle Drawn...")




#Player paddle
class PlayerPaddle(object):

    def __init__(self, screensize):

        self.screensize = screensize

        self.center_x = screensize[0] - 5
        self.center_y = int(screensize[1]*0.5)

        self.height = 100
        self.width = 10


        self.rect = pygame.Rect(0,self.center_y - int(self.height*0.5) , self.width, self.height)
        self.color = (255,255,100)

        self.speed = 6
        self.direction = 0

    def update(self):

        self.center_y += self.direction*self.speed

        self.rect.center = (self.center_x, self.center_y)

        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self.screensize[1]-1:
            self.rect.bottom = self.screensize[1]-1
        

    def render(self, screen):

        pygame.draw.rect(screen, self.color, self.rect, 0)
        pygame.draw.rect(screen, (0,0,0), self.rect, 1)
        #print("Player_paddle Drawn...")




def main():

    #start the pygame window
    pygame.init()

    #make screen
    screensize = (640,320)
    screen = pygame.display.set_mode(screensize)
    #make clock to control fps

    clock = pygame.time.Clock()
    pong = Pong(screensize)
    ai_paddle = AIPaddle(screensize)
    player_paddle = PlayerPaddle(screensize)
    running = True

    #while game is running
    while running:
        #FPS limiting phase
        clock.tick(60)

        #grab all the inputs
        #event handling phase
        for event in pygame.event.get():

            #check for quitting
            if event.type == QUIT:
                #pygame.quit()
                running = False

            if event.type == KEYDOWN:

                if event.key == K_UP:
                    player_paddle.direction = -1
                elif event.key == K_DOWN:
                    player_paddle.direction = 1

            if event.type == KEYUP:
                if event.key == K_UP and player_paddle.direction == -1:
                    player_paddle.direction = 0

                elif event.key == K_DOWN and player_paddle.direction == 1:
                    player_paddle.direction = 0

        #object update
        ai_paddle.update(pong)
        player_paddle.update()
        pong.update(player_paddle, ai_paddle)


        if pong.hit_edge_left:
            print("You Won")
            running = False

        elif pong.hit_edge_right:
            print("You Lost")
            running = False

        #rendering the game
        #make screen-1 black and working on screen-2
        screen.fill((100,100,100))


        #render the objects
        ai_paddle.render(screen)
        player_paddle.render(screen)
        pong.render(screen)

        pygame.display.update()
        #we need three objects , ball , human paddle, computer controleld paddle



    pygame.quit()



main()


