import pygame
import random
import numpy as np
import time
import matplotlib.pyplot as plt


'''SHAPE FORMATS'''

# here the list of S represents, all orientations of the S-tetromino 
S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]
 
Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]
 
I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]
 
O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]
 
J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]
 
L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]
 
T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, I, O, Z, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# SCORE = 0
# index 0 to 6 represent shape




'''”object” is a kind of placeholder, letting Python know you don't want to inherit the properties of some other class. You're making a class with the very basic rules, and your code will set up everything else.'''
class Piece(object):
    def __init__(self, x, y, shape, rotation=0):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = rotation
    
    def copy(self):
        return Piece(self.x, self.y, self.shape, self.rotation)



class Tetris():

    def __init__(self):
        # window screen
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 700
        # tetris game in window screen (playing area)
        self.TETRIS_BOX_WIDTH = 600  # needed 10 blocks horizontally <--> so 300//10 = 30 px(?) for each block 
        self.TETRIS_BOX_HEIGHT = 600 # same as above
        # block size
        self.BLOCK_SIZE = 30

        # top left of playing area
        self.TOP_LEFT_X = (self.SCREEN_WIDTH-self.TETRIS_BOX_WIDTH)//2
        self.TOP_LEFT_Y = (self.SCREEN_HEIGHT-self.TETRIS_BOX_HEIGHT)


        # SCORE = 0
        # index 0 to 6 represent shape
        self.win = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        pygame.display.set_caption('Tetris')

        
        self.locked_positions = {}
        self.grid = self.create_grid(self.locked_positions)
        self.change_piece = False
        self.run = True

        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()

        
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.27
        
        self.score = 0

        self.action_space = range(32)

    def get_initial_state(self):
        return self.get_state_of_the_grid(self.grid)

    def create_grid(self, locked_pos = {}):
        grid = [[(0, 0, 0) for _ in range(20)] for _ in range(20)]

        for i in range(len(grid)):# 20
            for j in range(len(grid[i])):# 10
                if((j,i) in locked_pos):
                    c = locked_pos[(j,i)]
                    grid[i][j] = c
        return grid


    def get_shape(self):    # S, I, O, Z, J, L, T
        shape_idx = np.random.choice(range(7), 1, p=[0.26, 0.0, 0.0, 0.18, 0.17, 0.25, 0.14])[0]
        return Piece(10, 0, shapes[shape_idx])


    def check_lost(self, positions):
        for pos in positions:
            x, y = pos
            if y < 1:
                return True
        
        return False


    def draw_grid(self, surface, grid):

        sx = self.TOP_LEFT_X
        sy = self.TOP_LEFT_Y

        for i in range(len(grid)):
            pygame.draw.line(surface, (128,128,128), (sx, sy+i*self.BLOCK_SIZE), (sx+self.TETRIS_BOX_WIDTH, sy+i*self.BLOCK_SIZE))
            for j in range(len(grid[i])):
                pygame.draw.line(surface, (128,128,128), (sx+j*self.BLOCK_SIZE, sy+i*self.BLOCK_SIZE), (sx+j*self.BLOCK_SIZE, sy+self.TETRIS_BOX_HEIGHT))


    def clear_rows(self, grid, locked):
        inc = 0
        for i in range(len(grid)-1, -1, -1):
            row = grid[i]
            if (0,0,0) not in row:
                inc += 1
                ind = i
                for j in range(len(row)):
                    try:
                        del locked[(j, i)]
                    except:
                        continue
        if inc > 0:
            # SCORE += inc*40
            for key in sorted(list(locked), key = lambda x: x[1])[::-1]:
                x, y = key
                if y < ind:
                    newKey = (x, y + inc)
                    locked[newKey] = locked.pop(key)
        
        return inc


    def draw_next_shape(self, shape, surface):
        font = pygame.font.SysFont('comicsans', 30)
        label = font.render('Next Shape', 1, (255, 255, 255))
        
        sx = self.TOP_LEFT_X + self.TETRIS_BOX_WIDTH + 50
        sy = self.TOP_LEFT_Y + self.TETRIS_BOX_HEIGHT/2 - 100
        format = shape.shape[shape.rotation % len(shape.shape)]

        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    pygame.draw.rect(surface, shape.color, (sx+j*self.BLOCK_SIZE, sy+i*self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))
        
        surface.blit(label, (sx+10, sy-30))


    def draw_window(self, surface, grid, score=0):
        surface.fill((0,0,0))

        pygame.font.init()
        font = pygame.font.SysFont('comicsans', 60)
        label = font.render('Tetris', 1, (255, 255, 255))

        surface.blit(label, (self.TOP_LEFT_X + self.TETRIS_BOX_WIDTH/2 - (label.get_width()/2), 30))

        
        font = pygame.font.SysFont('comicsans', 30)
        label = font.render('Score: '+str(score), 1, (255, 255, 255))
        
        sx = self.TOP_LEFT_X + self.TETRIS_BOX_WIDTH + 50
        sy = self.TOP_LEFT_Y + self.TETRIS_BOX_HEIGHT/2 - 100
        # format = shape.shape[shape.rotation % len(shape.shape)]

        surface.blit(label, (sx+20, sy+160))
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                pygame.draw.rect(surface, grid[i][j], (self.TOP_LEFT_X+j*self.BLOCK_SIZE, self.TOP_LEFT_Y+i*self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE), 0)
        
        pygame.draw.rect(surface, (255, 0, 0), (self.TOP_LEFT_X, self.TOP_LEFT_Y, self.TETRIS_BOX_WIDTH, self.TETRIS_BOX_HEIGHT), 4) 

        self.draw_grid(surface, grid)
        # pygame.display.update()


    def is_occupied(self, grid, pos):
        return not(grid[pos[0]][pos[1]] == (0,0,0)) 


    def show(self, accepted_pos):
        g = np.ones(shape=(20,20))
        for pos in accepted_pos:
            g[pos[1]][pos[0]] = 0

    def convert_shape_format(self, shape):
        positions = []
        format = shape.shape[shape.rotation%len(shape.shape)]
        for i, line in enumerate(format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    positions.append((shape.x + j, shape.y + i))

        for i, pos in enumerate(positions):
            positions[i] = (pos[0]-2, pos[1]-4)
        
        return positions


    def valid_space(self, shape, grid, ignore_piece=None):

        accepted_pos = [[(j, i) for j in range(20) if grid[i][j]==(0,0,0)] for i in range(20)]
        accepted_pos = [ j for sub in accepted_pos for j in sub]
        
        if not(ignore_piece is None):
            new_pos = self.convert_shape_format(ignore_piece)
            for each_coor in new_pos:
                accepted_pos.append(each_coor)

        formatted = self.convert_shape_format(shape)

        for pos in formatted:
            if pos not in accepted_pos:
                if pos[1] > -1:
                    return False
        return True


    def valid_actions(self, grid, shape):

        actions = []
        shape_coordinates = self.convert_shape_format(shape)
        print('rotation = ', shape.rotation)
        print('shape coordinates = ', shape_coordinates)
        
        for i in range(20):
            translated_piece = Piece(i, shape.y, shape.shape, shape.rotation) 
            if (self.valid_space(translated_piece, grid, shape)):
                actions.append(i)
            del(translated_piece)

        return actions 


    def allowed_actions(self, actions, shape):
    
        pos = shape.x
        step = -1
        action_space = [pos]
        idx = actions.index(pos)
        print(pos, idx, actions)
        for i in range(idx-1, -1, -1):
            # print('*'*5, i, actions[i])
            if actions[i]+1 == actions[i+1]:
                action_space.append(actions[i])
            else:
                break
        
        for i in range(idx+1, len(actions)):
            # print('*'*5, i, actions[i])
            if actions[i]-1 == actions[i-1]:
                action_space.append(actions[i])
            else:
                break
        
        action_space.sort()
        return action_space

    def get_state_of_the_grid(self, grid):
        state = np.array([np.zeros(len(grid[0])) for i in range(len(grid))])
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if not (grid[i][j]==(0,0,0)):
                    state[i][j] = 1
        return state


    def step(self, action):


        # while self.run:
        
        self.grid = self.create_grid(self.locked_positions)
        self.fall_time += self.clock.get_rawtime()
        self.clock.tick()

        if self.fall_time/1000 > self.fall_speed:
            self.fall_time = 0
            self.current_piece.y +=1 
            # if not(self.valid_space(self.current_piece, self.grid)) and self.current_piece.y > 0:
            #     self.current_piece.y -=1
            #     self.change_piece = True

        empty_slots = self.allowed_actions(self.valid_actions(self.grid, self.current_piece), self.current_piece)
        print('empty_slots = ',empty_slots)
            
        prev_pos = (self.current_piece.x, self.current_piece.y, self.current_piece.shape, self.current_piece.rotation)
        
        for pos in self.convert_shape_format(self.current_piece):
            if pos[1]<0:
                action = 30

        if action in range(20):
            self.current_piece.x = action
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.x = prev_pos[0]
                self.current_piece.y = prev_pos[1]
                self.current_piece.shape = prev_pos[2]
                self.current_piece.rotation = prev_pos[3]
        elif action in range(20, 24):
            self.current_piece.rotation += (action-20)
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.rotation -= (action-20)
        elif action == 25:
            self.current_piece.x-=1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.x+=1
        elif action == 26:
            self.current_piece.x+=1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.x-=1
        elif action == 27:
            self.current_piece.y+=1
            if not(self.valid_space(self.current_piece, self.grid)):
                self.current_piece.y-=1
        else:
            pass # NO ACTION = NULL
            
        if self.need_to_lock(self.current_piece, self.grid):
            self.change_piece = True
        else:
            self.change_piece = False
            
        shape_pos = self.convert_shape_format(self.current_piece)
        print('shape_pos = ', shape_pos)
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                print(y, x)
                self.grid[y][x] =  self.current_piece.color
            
        if self.change_piece:
            print('='*40)
            for pos in shape_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.current_piece.color
            self.current_piece = self.next_piece
            self.next_piece = self.get_shape()
            self.change_piece = False
            self.score += 10*self.clear_rows(self.grid, self.locked_positions)


        # self.draw_window(self.win, self.grid, self.score)
        # self.draw_next_shape(self.next_piece, self.win)
        # pygame.display.update()
        if self.check_lost(self.locked_positions):
            self.run = False
        return self.get_state_of_the_grid(self.grid)

    def need_to_lock(self, current_piece, grid):
        piece_positions = self.convert_shape_format(current_piece)
        at_start = False

        if piece_positions is None:
            return False

        for pos in piece_positions:
            at_start = at_start or pos[1]<0

        if at_start:
            return False
        

        col_set = {}
        for pos in piece_positions:
            if pos[0] in col_set.keys():
                if col_set[pos[0]] < pos[1]:
                    col_set[pos[0]] = pos[1]
            else:
                col_set[pos[0]] = pos[1]
        

        down_pos = []
        for k,v in col_set.items():
            down_pos.append((k, v+1))

        
        for pos in down_pos:
            if pos[1]==20:
                return True

        empty_down = True
        for pos in down_pos:
            empty_down = empty_down and pos[1] in range(20) and pos[0] in range(20) and grid[pos[1]][pos[0]] == (0,0,0)
            
        
        return not(empty_down)

    def close(self):
        pygame.display.quit()


# env = Tetris()
# # print("ksvjbuoekbgkw")
# while(env.run):
#     action = int(input())
#     env.step(action)