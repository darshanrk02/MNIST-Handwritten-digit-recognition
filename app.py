import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2


# initializing pygame module
pygame.init()

# defining window size
WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDARYINC = 5

# initializing colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

# to save image, initially initialized to false
IMAGESAVE = False

# loading the DL model
MODEL = load_model("bestmodel.h5")

LABELS = {0:"Zero", 1:"One", 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

# caption for our window
pygame.display.set_caption("Digit Board")

DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

FONT = pygame.font.Font('freesansbold.ttf', 18)

iswriting = False

# creating lists to store x and y co-ordinates
number_xcord = []
number_ycord = []

image_cnt = 1

PREDICT = True

# to retain display window
while True:
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)
            
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        # to cpature digit that is written on display surface
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            # to draw rectangle around the digit written
            rect_min_x, rect_max_x = max(number_xcord[0]-BOUNDARYINC,0), min(WINDOWSIZEX, number_xcord[-1]+BOUNDARYINC)

            rect_min_y, rect_max_y = max(number_ycord[0]-BOUNDARYINC,0), min(WINDOWSIZEX, number_ycord[-1]+BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            # saving the image
            if IMAGESAVE:
                cv2.imwrite('image.png')
                image_cnt += 1

            # predicting the digit using model
            if PREDICT:
                
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])  

                # Create a rectangle object
                rect = pygame.Rect(rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y)

                # Draw the rectangle on the display surface
                pygame.draw.rect(DISPLAYSURFACE, RED, rect, 1)
                
                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURFACE.blit(textSurface, textRecObj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURFACE.fill(BLACK)  

    pygame.display.update()
    







