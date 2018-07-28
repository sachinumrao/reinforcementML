import pygame
from pygame.locals import *

class Pong(object):

    def __init__(self, screensize):

        self.screensize = screensize

        self.centerx = int(screensize[0]*0.5)
        self.centery = int(screensize[1]*0.5)
        self.radius = 5
        self.rect = pygame.Rect(self.centerx-self.radius,
                                self.centery-self.radius,
                                self.radius**2, self.radius**2)

        self.color = (100,100,255)

        self.direction = [1,1]

        self.speed = 5

        self.hit_edge_left = False
        self.hit_edge_right = False


    def update(self, player_paddle=None, ai_paddle=None):
        
        self.centerx += self.direction[0]*self.speed
        self.centery += self.direction[1]*self.speed

        self.rect.center = (self.centerx, self.centery)

        if self.rect.top <=0:
            self.direction[1] = 1

        elif self.rect.bottom >= self.screensize[1]-1:
            self.direction[1] = -1


        if self.rect.right >= self.screensize[0]-1:
            self.hit_edge_right = True

        elif self.rect.left <= 0:
            self.hit_edge_left = True


    def render(self,screen):
        pygame.draw.circle(screen, self.color, self.rect.center, self.radius, 0)


class Ai_Paddle(object):

    def __init__(self,screensize):

        self.screensize = screensize
        self.centerx = int(5)
        self.centery = int(screensize[1]*0.5)
        self.height= 100
        self.width = 10
        self.speed = 2
        self.rect = pygame.Rect(0,self.centery - int(self.height *0.5), self.width, self.height)
        self.color = (255,0,0)

    def update(self, pong):

        if pong.rect.top < self.rect.top:
            self.centery -= self.speed
        elif pong.rect.bottom > self.rect.bottom:
            self.centery += self.speed

        self.rect.center = (self.centerx, self.centery)

    def render(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 0)
        pygame.draw.rect(screen , (0,0,0) , self.rect, 1)


class Player(object):

    def __init__(self, screensize):

        self.screensize = screensize
        self.centerx = screensize[0]-5
        self.centery = int(screensize[1]*0.5)
        self.height = 100
        self.width = 10
        self.rect = pygame.Rect(0, self.centery-int(self.height*0.5), self.width, self.height)
        self.color = (100,255,100)
        self.speed = 3
        self.direction = 0



    def update(self, pong):

        self.centery += self.direction*self.speed
        self.rect.center = (self.centerx, self.centery)

    def render(self, screen):
        pygame.draw.rect(screen,  self.color, self.rect, 0)
        pygame.draw.rect(screen, (0,0,0), self.rect, 1)




def main():
    pygame.init()

    screensize = (640,480)
    screen = pygame.display.set_mode((640,480))
    clock = pygame.time.Clock()
    running = True

    pong = Pong(screensize)

    ai_paddle = Ai_Paddle(screensize)

    pl_paddle = Player(screensize)

    while running:
        clock.tick(10)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                running = False

            if event.type == pygame.KEYDOWN:
                print("button pressed")
                if event.key == pygame.K_UP :
                    pl_paddle.direction = -1
                elif event.key == pygame.K_DOWN :
                    pl_paddle.direction = 1

            if event.type == pygame.KEYUP :
                if event.key == pygame.K_UP :
                    if evnet.key == pygame.K_UP and pl_paddle.direction == -1:
                        pl_paddle.direction = 0

                    elif evnet.key == pygame.K_DOWN and pl_paddle.direction == -1:
                        pl_paddle.direction = 0

        ai_paddle.update(pong)
        pl_paddle.update(pong)
        pong.update(pl_paddle, ai_paddle)

        if pong.hit_edge_left:
            print("You Win")
            running = False

        elif pong.hit_edge_right:
            print("you lost")

        screen.fill((100,100,100))

        pong.render(screen)

        ai_paddle.render(screen)
        pl_paddle.render(screen)
        pygame.display.flip()

main()