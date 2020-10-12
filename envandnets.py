# coding: utf-8

# # Imports and Setup

# In[1]:
 
#https://www.youtube.com/watch?v=gsufHS-bqxA

# http://pytorch.org/
from __future__ import print_function
from __future__ import division
import os
from os import path
goog = False

#SET THESE TO FALSE IF JUST WANNA IMPORT ENV ETC
wantToTrain = False
wantToTest = False

if(goog):
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

    accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

    get_ipython().system('pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.0-{platform}-linux_x86_64.whl torchvision')
    get_ipython().system('pip install imageio')
    get_ipython().system('pip install setproctitle')
    get_ipython().system('pip install Pillow --upgrade')

import torch
import torch.multiprocessing as mp
gpu = True
if(gpu and __name__  ==  '__main__'):
    mp.set_start_method('spawn')
    mp.freeze_support()

from PIL import Image
import imageio


# In[2]:


import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


# In[3]:


if(goog):
    get_ipython().system('pip install -U -q PyDrive')
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    import datetime


# In[4]:


if(goog):
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)


# # Environment

# In[5]:

plotstuff = False


#by convention, empty and agent are 0 and 1

'''
    imnames = ["empty",
    "agent", #index 0 in our tensor
    "greenball", #transformable0
    "greenguy", #transformable1
    #"box", #transformable2
    #"carrot", #transformable3
    #"coffee", #transformable4
    #"fries", #transformable5
    "blueball", #transformed0
    "yellowguy", #transformed1
    #"hat", #transformed2
    #"orange", #transformed3
    #"icecream", #transformed4
    #"pizza", #transformed5
    "tv", #untransformable
    "heart", #untransformable
    "enemy",
    "block",
    "water"]
'''

imnames = ["empty",
"agent", #index 0 in our tensor
"greenball", #transformable0
#"greenguy", #transformable1
"blueball", #transformed0
#"yellowguy", #transformed1
"tv", #untransformable
"heart", #untransformable
"enemy",
"block",
"water"]

#change values depending on imnames
transformable_0_idx = 1
transformed_last_idx = 2
untransformable_0_idx = 3
untransformable_last_idx = 4

numTransformable = int((transformed_last_idx - transformable_0_idx + 1)/2)
numUntransformable = untransformable_last_idx - untransformable_0_idx + 1
enemy_idx = untransformable_last_idx + 1
block_idx = untransformable_last_idx + 2
water_idx = untransformable_last_idx + 3

if(goog):
    imIds = ["1ClnO_S5mI9uxzBt9SgUeCSnEilr9-vTy",
            "1FYpQM6gfVqSOAiQjQ4uuPehB36Chnz2W",
            "1WDIynyzj3jFCKDY1fn10DnpAxLwYa2nI",
            #"1WbUX1U6M6PxpxJIf4Pr8SZBYvcE_BsXx",
            "1ljQYOcGw7qATGYaPamKYVY7OhjL6Nv1x",
            #"1Lmphg8LSSwUh0iMll68QfBoArvVMsVd3",
            "1fL5veat2ShZFPHl0rHPuCzp4ypLMV8hf",
            "1mSFdUGGP5GfYZPV9FybC6_f7dr-IVF2J",
            "17_5NdnzzIOMBhc_yJPaqyIJoW8sTTmo_",
            "1Sv3gkg0JQJ26rDOagvEYuV32QXdmFJtm",
            "1aQyn2tV26a4hrFZ3LRkBOroNHrNE8yAW"]

    for i in range(len(imnames)):
      file_id = imIds[i]
      downloaded = drive.CreateFile({'id': file_id})
      downloaded.GetContentFile(imnames[i]+'.jpg')

    #add the grey background for the instructions to live in
    downloaded = drive.CreateFile({'id': '1kYEhhyZF6ENe5slU9eEIyRj2niOxTiJI'})
    downloaded.GetContentFile('grey.jpg')


# In[6]:


#workaround to avoid register_extensions PIL error
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions


# In[7]:


mode = "nolstm" #set your mode! Will affect all file names


# In[8]:
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch
import os,sys
import datetime
import imageio
from pprint import pprint
import time
import datetime

####################################################################
########################## SETUP ###################################
####################################################################
#~~~~~~~~~~~ FUNDAMENTAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#all images have to be 32x32
blockSide = 32
#the grid is 10x10 blocks
grid_n = 6

#~~~~~~~~~~~ IMAGES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tilesDir = "tiles/"

#there should be 18 elements + 1 "empty" cell
numCellTypes = len(imnames)-1
grid_depth = numCellTypes
assert(grid_depth==8)
episode_max_length = 300

act_dim = 13

#~~~~~~~~~~~~ Mapping b/t actions, subtasks and integers
act_types = ["Move ", "Pick up ", "Transform "]
act_dirs = ["N", "S", "W", "E"]

def actionNumToText(action):
  if action==12 or not isinstance(action, int):
    s = "None"
  elif(action < 0 or action > 12):
    s = "Invalid action"
  else:
    act_type = int(action/4) #0 move, 1 pickup, 2 transform
    act_dir = action - 4*act_type #0-3 is NSWE
    s = act_types[act_type] + act_dirs[act_dir]
  return s

#1s are the verbs, 2s are the objects
subtask_1s = ["Visited ", "Picked up ", "Transformed "] #, "Picked up all", "Transformed all"] #will deal with "all" later
subtask_2s = imnames[transformable_0_idx+1:untransformable_last_idx+2]
subtask_1s_num = len(subtask_1s)
subtask_2s_num = len(subtask_2s)


#~~~~~~~~~~~~ CLEANUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def softremove(path):
  try:
    os.remove(path)
    return
  except Exception as e:
    return

for fname in os.listdir('./'):
    if fname.startswith(mode):
        softremove(os.path.join('.', fname))

####################################################################
################## IMPORT IMAGES ###################################
####################################################################
#print(imnames.index("hat")) #to get the reverse mapping objectname->index

assert(len(imnames)==numCellTypes+1)

imlist = []
for i in range(numCellTypes+1):
  imlist.append(Image.open(tilesDir + imnames[i] + ".jpg"))
emptyim = imlist[0]
agent = imlist[1]

greyimg = Image.open(tilesDir + "grey.jpg")

####################################################################
################## MAZEWORLD CLASS #################################
####################################################################
class MazeWorld(object):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #initialize the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, grid_n, blockSide, numCellTypes, instruction, maxnumsteps, reset_total_count, gpu=False, useEnemy=False, changegrid=False):
    self.grid_n = grid_n
    self.blockSide = blockSide
    self.numCellTypes = numCellTypes
    self.instruction = instruction
    self.originalinstruction = instruction
    assert(type(instruction)==type([]))
    self.maxnumsteps = maxnumsteps
    self.numsteps = 0
    self.changegrid = changegrid
    self.gpu = gpu
    self.done = False

    #initialize the grid
    self.initgrid = self.buildgrid(grid_n, instruction)

    #place agent
    #CHAO EDIT THIS PLS
    while True:
      i = int(grid_n * np.random.uniform())
      j = int(grid_n * np.random.uniform())
      if(self.initgrid[i][j] < block_idx+1):
        break
    self.agent_i = i
    self.agent_j = j
    self.initagent_i = i
    self.initagent_j = j
    #print("initially i and j are " + str(i) + "and" + str(j))

    #make tensor
    self.t = self.initgridToTensor()
    if(gpu):
        self.t = self.t.cuda()

    #variables for enemy
    self.useEnemy = useEnemy
    self.enemy_lifelength = 10 #how many steps it stays
    self.enemy_prob = .05 #prob at each step of appearing
    self.enemy_age = -1
    self.enemy_i = -1
    self.enemy_j = -1
    
    #keep track of how many times we have reset
    self.reset_total_count = reset_total_count
    self.reset_current_count = 0
    
  def reset(self, instructions=None, easy = False, hard = False):
    if(instructions!=None):
      self.instruction = instructions
      self.originalinstruction = instructions
    else:
      self.instruction = self.originalinstruction
    self.numsteps = 0
    
    assert(not (easy and hard))
    
    
    for ins in self.instruction:
      lastword = ins.split()[-1]
      objIndex = imnames.index(lastword)
        
    if(self.changegrid or instructions!=None):
      #CHAO EDIT THIS PLS
      if(self.reset_current_count >= self.reset_total_count):

          #initialize the grid
          self.initgrid = self.buildgrid(grid_n, self.originalinstruction)
          self.initgrid[self.agent_i][self.agent_j] = 0

          #place agent
          for itr in range(1000):
            i = int(grid_n * np.random.uniform())
            j = int(grid_n * np.random.uniform())
            min_dist = float("Inf")
            max_dist = 0
            if(self.initgrid[i][j] < block_idx+1):
                ### add constrants for the env
                if easy:
                    for col in range(grid_n):
                        for row in range(grid_n):
                            if self.initgrid[col][row] == objIndex:
                                temp_dist = (np.abs(col-i) + np.abs(row-j))
                                if temp_dist > max_dist:
                                    max_dist = temp_dist

                    if max_dist <= 3:
                        break
                    
                elif hard:
                    for col in range(grid_n):
                        for row in range(grid_n):
                            if self.initgrid[col][row] == objIndex:
                                temp_dist = (np.abs(col-i) + np.abs(row-j))
                                if temp_dist < min_dist:
                                    min_dist = temp_dist

                    if min_dist > 3:
                        break
                else:
                    break
                        
            if itr == 999:
                if(easy):
                    print ("Resetting with easy constraint not working after 1k tries, trying again")
                elif(hard):
                    print ("Resetting with hard constraint not working after 1k tries, trying again")
                else:
                    print("Resetting with no constraint not working after 1k tries, trying again")
                self.reset()

          self.agent_i = i
          self.agent_j = j
            
          self.reset_current_count = 0
    else:
      self.agent_i = self.initagent_i
      self.agent_j = self.initagent_j
      #print("Resetting; now i and j are " + str(self.agent_i) + "and" + str(self.agent_j))

    #make tensor
    self.t = self.initgridToTensor()
    if(self.gpu):
        self.t = self.t.cuda()

    #variables for enemy
    #self.useEnemy stays the same
    self.enemy_age = -1
    self.enemy_i = -1
    self.enemy_j = -1
    
    self.done = False
    
    self.reset_current_count += 1
    
    return self.t
    
  #sambuildgrid
  def buildgrid(self, grid_n, instruction):
    #some values to support random env generation (assumes 18 objs)
    prob_nonempty = .3 #prob that a cell is not empty
    prob_water_ifnonempty = .25
    prob_block_ifnonemptynonwater = .45
    prob_untransformable_ifnothingelse = numUntransformable/(numUntransformable+numTransformable)

    grid = np.zeros((grid_n,grid_n), dtype=int)

    #make sure objects in instructions exist in the grid!
    for ins in instruction:
      lastword = ins.split()[-1]
      objIndex = imnames.index(lastword)

      while True:
        i = int(grid_n * np.random.uniform())
        j = int(grid_n * np.random.uniform())
        if(grid[i][j] == 0):
          grid[i][j] = objIndex
          break

    for i in range(grid_n):
      for j in range(grid_n):
        if(np.random.uniform()<prob_nonempty and grid[i][j]==0):
          #square is nonempty
          if(np.random.uniform()<prob_water_ifnonempty):
            grid[i][j] = water_idx+1 # water
          else:
            if(np.random.uniform()<prob_block_ifnonemptynonwater):
              grid[i][j] = block_idx+1 #block
            else:
              if(np.random.uniform()<prob_untransformable_ifnothingelse):
                grid[i][j] = 1+untransformable_0_idx + int(numTransformable*np.random.uniform())
                #nontransformable
              else:
                grid[i][j] = 1+transformable_0_idx+int(numTransformable*np.random.uniform())
                #o/w pick transformable

    return grid

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #act in the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def step(self, action):
    if(self.done):
        raise ValueError('Trying to step in a done env.')
    
    reward = -.1
    done = False
    taskupdate = ""

    #execute action
    if(action != 12): #12 is no op
      act_type = int(action/4) #0 move, 1 pickup, 2 transform
      act_dir = action - 4*act_type #0-3 is NSWE

      #get the cell the action is "acting on"
      act_i = self.agent_i
      act_j = self.agent_j
      if(act_dir==0):
        act_i -= 1
      elif(act_dir==1):
        act_i += 1
      elif(act_dir==2):
        act_j -= 1
      else: #(act_dir==1)
        act_j += 1

      #check that the "acted on" cell is not OOB or a block
      if(not(act_j <0 or act_i <0         or act_j>=self.grid_n or act_i>=self.grid_n)):
        if(self.t[block_idx][act_i][act_j] != 1):
          #deal with action
          if(act_type==0): #move
            self.t[0][act_i][act_j] = 1
            self.t[0][self.agent_i][self.agent_j] = 0
            self.agent_i = act_i
            self.agent_j = act_j

            #if we visited an object, change taskupdate
              #start by finding out what object is there
            objidx = -1
            for idx in range(1, self.numCellTypes-1):
              if(self.t[idx][act_i][act_j]==1):
                objidx = idx
                break
            if(objidx!=-1):
              taskupdate = "Visited " + imnames[objidx+1]
          else: #pick up or transform
            if(act_type==1):
              action_name = "Picked up "
            elif(act_type==2):
              action_name = "Transformed "
            
            #start by finding out what object is there
            objidx = -1
            for idx in range(1, self.numCellTypes):
              if(self.t[idx][act_i][act_j]==1):
                objidx = idx
                break

            #we know object and where it is, let's execute
            if(objidx != -1 and objidx != water_idx):
              didSomething = False

              #if transformable and transform action is taken,
              #we need to "replace" object with its transformation
              if(objidx >= transformable_0_idx and objidx < transformable_0_idx+numTransformable):
                if(act_type==2):
                  self.t[objidx+numTransformable][act_i][act_j]=1
                  self.t[objidx][act_i][act_j]=0 #remove object
                  didSomething = True

              #LATER ADDITION: DECIDED TO MAKE TV TRANSFORM TO HEART
              if(objidx ==  untransformable_last_idx-1 and act_type==2):
                  self.t[untransformable_last_idx][act_i][act_j]=1
                  self.t[objidx][act_i][act_j]=0 #remove object
                  didSomething = True
                
              if(act_type==1):
                self.t[objidx][act_i][act_j]=0 #remove object                
                didSomething = True

              if(objidx==enemy_idx): #enemy
                self.enemy_age = 1 #so we can clean up
                if(act_type==2): #transform
                  reward += .9
                didSomething = True

              if(didSomething):
                taskupdate = action_name + imnames[objidx+1]


    #deal with enemy
    if(self.useEnemy):
      if(self.enemy_age==-1):
        if(self.enemy_prob > np.random.uniform()):
          #create enemy
          countloop = 0
          while True:
            self.enemy_i = int(grid_n * np.random.uniform())
            self.enemy_j = int(grid_n * np.random.uniform())
            fibre = self.t[:,self.enemy_i,self.enemy_j]
            if(int(sum(fibre)) == 0):
              self.t[enemy_idx][self.enemy_i][self.enemy_j] = 1
              self.enemy_age = self.enemy_lifelength
              break
            countloop += 1
            if(countloop==100):
              print("Looping too many times for enemy creation!")
              break
      else:
        self.enemy_age -= 1
        if(self.enemy_age==0): #enemy's life has ended
          self.enemy_age = -1
          self.t[enemy_idx][self.enemy_i][self.enemy_j] = 0
          self.enemy_i = -1
          self.enemy_j = -1
        else: #move the enemy randomly
          #generate the 4 possible cells the agent could move to
          dirs = [[self.enemy_i, self.enemy_j] for x in range(4)]
          dirs[0][1] += 1
          dirs[1][1] -= 1
          dirs[2][0] += 1
          dirs[3][0] -= 1

          legaldirs = []

          for d in dirs:
            if(d[0]>=0 and d[0] < self.grid_n and             d[1]>=0 and d[1] < self.grid_n):
              fibre = self.t[:,d[0],d[1]]
              if(int(sum(fibre)) == 0):
                legaldirs.append(d)

          if(len(legaldirs)!=0):
            self.t[enemy_idx][self.enemy_i][self.enemy_j] = 0
            newdir = random.choice(legaldirs)
            self.enemy_i = newdir[0]
            self.enemy_j = newdir[1]
            self.t[enemy_idx][self.enemy_i][self.enemy_j] = 1

    #if in water, punish!
    if(self.t[water_idx][self.agent_i][self.agent_j] == 1):
      reward -= .3

    #reward + 1 and terminate if instruction is done
    if(self.instruction[0] == taskupdate):
      self.instruction = self.instruction[1:]
      if(len(self.instruction)==0):
        done = True
        self.done = True
        reward += 1

    #increment numsteps and terminate if numsteps > maxnumsteps
    self.numsteps += 1
    if(self.numsteps > self.maxnumsteps):
      done = True

    return self.t, reward, done, taskupdate

  #load a tensor, throws an error if exists enemy
  def loadTensor(self, newT, newInstruction):
    #set tensor
    self.t = newT
    #set agent position
    foundAgent = False
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        if(self.t[0][i][j] == 1):
          foundAgent = True
          self.agent_i = i
          self.agent_j = j
          
    assert(foundAgent) #if no agent in grid something is wrong
    
    #throw error if there is an enemy
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        if(self.t[enemy_idx][i][j] == 1):
          raise ValueError('Code cannot handle loading grid with agent yet.')
    self.useEnemy = False
    
    #initgrid is none because load could be from any initgrid
    self.initgrid = False
    
    #reload instruction, set initinstruction to none for same reason as above
    self.originalinstruction = None
    self.instruction = newInstruction
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #render the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def makeImageFromGrid(self, gridarr, step, action, reward, taskupdate, cumreward):
    assert(self.grid_n == gridarr.shape[1])

    result_width = int(self.blockSide*self.grid_n*1.5)
    result_height = self.blockSide*(self.grid_n+1)
    result = Image.new('RGB', (result_width, result_height))

    #DRAW GRID
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        itemindex = gridarr[j][i]
        result.paste(im=imlist[itemindex], box=(i*self.blockSide, j*self.blockSide))

    #DRAW GREY AREA FOR INSTRUCTIONS
    for i in range(self.grid_n, int(self.grid_n*1.5)):
      for j in range(self.grid_n):
        result.paste(im=greyimg, box=(i*self.blockSide, j*self.blockSide))

    #WRITING AND LINES
    #font_type = ImageFont.truetype("arial.ttf", 12, encoding="unic")
    draw = ImageDraw.Draw(result)
      #bottom line
    draw.line(xy=[0, self.blockSide*self.grid_n, self.blockSide*self.grid_n,      self.blockSide*self.grid_n], fill=(255,255,255))
      #bottom text
    mytext1 = "Step: " + str(step) + ", act: " + actionNumToText(action) + ", rew: " +       str(reward) + ", cumrew = " + str(cumreward+reward)
    mytext2 = "Taskupdate: " + taskupdate
    draw.text(xy =(0, self.blockSide*(self.grid_n)),       text=mytext1, fill=(255,255,255))
    #, font=font_type)
    draw.text(xy =(0, self.blockSide*(self.grid_n+.5)),       text=mytext2, fill=(255,255,255))
              #, font=font_type)
      #instruction text
    num_instructions = len(self.originalinstruction)
    current_instruction = num_instructions - len(self.instruction)
    for i in range(num_instructions):
      if(i!=current_instruction):
        draw.text(xy = (self.grid_n*self.blockSide, i*self.blockSide/3),           text=self.originalinstruction[i], fill=(0,0,0))#, font = font_type)
      else:
        draw.text(xy = (self.grid_n*self.blockSide, i*self.blockSide/3),           text=self.originalinstruction[i], fill=255)#, font = font_type)

    return result

  def makeTextImage(self, mytext):
    result_width = int(self.blockSide*self.grid_n*1.5)
    result_height = self.blockSide*(self.grid_n+1)
    result = Image.new('RGB', (result_width, result_height))
    #font_type = ImageFont.truetype("arial.ttf", 32, encoding="unic")
    draw = ImageDraw.Draw(result)
    draw.text(xy =(result_width*.4, result_height*.4),       text=mytext, fill=(255,255,255))#, font=font_type)
    return result

  def render(self, filename, step, action, reward, taskupdate, cumreward):
    #place agent
    newgrid, self.agent_i, self.agent_j = self.tensorToArr()
    newgrid[self.agent_i][self.agent_j] = 1

    result = self.makeImageFromGrid(newgrid, step, action, reward,     taskupdate, cumreward)
    outfile = filename + ".jpg"
    #result.show()
    result.save(outfile)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #switch b/t representations of the world~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def initgridToTensor(self):
    t = torch.zeros((self.numCellTypes,self.grid_n,self.grid_n))#, dtype=torch.int32)

    for i in range(self.grid_n):
      for j in range(self.grid_n):
        idx = self.initgrid[i][j]
        if(idx!=0):
          t[idx-1][i][j] = 1

    t[0][self.agent_i][self.agent_j] = 1 #place agent

    return t

  def tensorToArr(self):
    arr = np.zeros((self.grid_n,self.grid_n), dtype=int)

    for ct in range(self.numCellTypes):
      for i in range(self.grid_n):
        for j in range(self.grid_n):
          if(self.t[ct][i][j] == 1):
            if(ct == 0):
              agent_i = i
              agent_j = j
            else:
              arr[i][j] = ct+1
    return arr, agent_i, agent_j


# # ENV TESTING (IGNORE)

# In[9]:


envtest = False

if(envtest):

    ## BASIC MAIN WITH RANDOM ACTION, commented out to avoid unnecessary execution

    '''
    #"instructions =" -> change instructions
    #See all the booleans in the first subsection
    #numiter first variable -> max number of steps per episode
    ####################################################################
    ######################## MAIN LOOP #################################
    ####################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                          Some basic settings
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    numiter = 10

    logging = True #save states, actions, rewards
    makeimages = True #make an image for each step
    makegif = True #make a gif for the whole episode
    printingresults = True #print action, reward, taskupdate every step
    justprintsummary = False #only print numsteps and cumreward @ the end

    instructions = ["Picked up greenguy"]#,
      #  "Picked up blueball"] #etc.
    #Picked up, Transformed, Visited

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                          Init the Env
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    env = MazeWorld(grid_n,blockSide,numCellTypes, \
      instructions, \
      numiter-1)
    if(makeimages or makegif):
      env.render(mode+"0", 0, "none", 0, "Start", 0)
    if(logging):
      envlog = torch.zeros((numiter+1,env.numCellTypes,env.grid_n,env.grid_n))#,\
       #dtype=torch.int32)
      envlog[0] = env.t
      actRewDoneLog = []

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                          Run an episode
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    done = False
    i = 1
    cumreward = 0
    while not done:
      action = random.randint(0,12)
      obs, reward, done, taskupdate = env.step(int(action))
      renderfilename = mode + str(i)

      if(logging):
        envlog[i] = env.t
        actRewDoneLog.append([action, reward, done])

      if(printingresults):
        print(str(i) + ": Action was " + str(action) + \
          ", reward is " + str(reward) + ", taskupdate is " \
          + taskupdate + ".")

      if(makeimages or makegif):
        env.render(renderfilename, i, action, reward, taskupdate, cumreward)

      i += 1
      cumreward += reward

    if(printingresults or justprintsummary):
      print("Env terminated after " + str(i-1) + " steps, cumreward = " + str(cumreward))

    if(logging):
      softremove('envlog.tensor')
      torch.save(envlog, 'envlog.tensor')

    if(makegif):
      #code from Alex Buzunov at https://goo.gl/g2G9c6
      duration = 0.5
      filenames = sorted(filter(os.path.isfile, [x for x in os.listdir() \
      if (x.endswith(".jpg") and x.startswith(mode))]), key=lambda \
      p: os.path.exists(p) and os.stat(p).st_mtime or \
      time.mktime(datetime.now().timetuple()))

      #make an image that says "START"
      result = env.makeTextImage("START")
      result.save(mode + "Start.jpg")

      #make an image that says "END"
      result = env.makeTextImage("END")
      result.save(mode + "End.jpg")
      filenames.append(mode+"End.jpg")

      images = []
      images.append(imageio.imread(mode + "Start.jpg"))
      for filename in filenames:
        images.append(imageio.imread(filename))

      #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
      output_file = mode + '.gif'
      imageio.mimsave(output_file, images, duration=duration)
      '''


# In[10]:


#EXPORT EPISODE TO DRIVE, also commented out
'''
uploadIndividualImages = False #otherwise just upload gif
if(uploadIndividualImages):
  for filename in filenames:
    f = drive.CreateFile()
    f.SetContentFile(filename) #add the contents of your local file to the remove file
    f.Upload() #create your remote file

f = drive.CreateFile()
f.SetContentFile(mode+'.gif')
f.Upload()
'''


# In[11]:


if(envtest):
    #MAKE AND PLOT ENV

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import cv2

    env = MazeWorld(grid_n,blockSide,numCellTypes,           ["Picked up blueball"],           100, changegrid=True)
    env.render("step0", 0, "none", 0, "Start", 0)

    img=cv2.imread('step0.jpg')
    plt.axis("off")
    imgplot = plt.imshow(img)


# In[12]:


if(envtest):
    env.reset()
    env.render("test0", 0, "none", 0, "Start", 0)

    img=cv2.imread('test0.jpg')
    plt.axis("off")
    imgplot = plt.imshow(img)
    plt.show()


# In[13]:


if(envtest):
    #EPISODE DIAGNOSTICS (MANUAL ACTION)

    #set lilmode and clear all related files
    lilmode = "samtest"
    for fname in os.listdir('./'):
        if fname.startswith(lilmode):
            softremove(os.path.join('.', fname))

    state = env.reset()

    plt.axis("off")
    renderfilename = lilmode+"0"
    env.render(renderfilename, 0, "none", 0, "Start", 0)
    img=cv2.imread(renderfilename+".jpg")
    imgplot = plt.imshow(img)
    plt.show()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                          Run an episode
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    done = False
    i = 1
    cumreward = 0

    while not done:
      action = int(input("Choose action pls:"))

      state, reward, done, taskupdate = env.step(action)

      actText = actionNumToText(action)

      print(str(i) + ": Action was " + actText +         ", reward is " + str(reward) + ", taskupdate is "         + taskupdate + ".")

      #render
      renderfilename = lilmode + str(i)
      env.render(renderfilename, i, action, reward, taskupdate, cumreward)
      img=cv2.imread(renderfilename+".jpg")
      plt.axis("off")
      imgplot = plt.imshow(img)
      plt.show()

      i += 1
      cumreward += reward

    print("Env terminated after " + str(i-1) + " steps, cumreward = " + str(cumreward))

    #make the gif
    #code from Alex Buzunov at https://goo.gl/g2G9c6
    duration = 1
    filenames = sorted(filter(os.path.isfile, [x for x in os.listdir()     if (x.endswith(".jpg") and x.startswith(lilmode))]), key=lambda     p: os.path.exists(p) and os.stat(p).st_mtime or     time.mktime(datetime.now().timetuple()))

    #make an image that says "START"
    result = env.makeTextImage("START")
    result.save(lilmode + "Start.jpg")

    #make an image that says "END"
    result = env.makeTextImage("END")
    result.save(lilmode + "End.jpg")
    filenames.append(lilmode+"End.jpg")

    images = []
    images.append(imageio.imread(lilmode + "Start.jpg"))
    for filename in filenames:
      images.append(imageio.imread(filename))

    #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    output_file = lilmode + '.gif'
    imageio.mimsave(output_file, images, duration=duration)

    #EXPORT (PICS AND) GIF TO DRIVE
    uploadIndividualImages = False #otherwise just upload gif
    if(uploadIndividualImages):
      for filename in filenames:
        f = drive.CreateFile()
        f.SetContentFile(filename) #add the contents of your local file to the remove file
        f.Upload() #create your remote file

    f = drive.CreateFile()
    f.SetContentFile(lilmode+'.gif')
    f.Upload()


# # A3C training
# Source: https://github.com/dgriff777/rl_a3c_pytorch with some modifications

# In[14]:

# In[15]:


#shared_optim.py with no modifications
#OPTIMIZER


import math
import torch
import torch.optim as optim
from collections import defaultdict

 
class SharedRMSprop(optim.Optimizer):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=7e-4,
                 alpha=0.99,
                 eps=0.1,
                 weight_decay=0.0005,
                 momentum=0,
                 centered=False):
        defaults = defaultdict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered)
        super(SharedRMSprop, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['grad_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['momentum_buffer'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'].share_memory_()
                state['step'].share_memory_()
                state['grad_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg,
                                             grad_avg).sqrt().add_(
                                                 group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss


class SharedAdam(optim.Optimizer):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-3,
                 weight_decay=0,
                 amsgrad=False):
        defaults = defaultdict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad)
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
                state['max_exp_avg_sq'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till
                    # now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step'].item()
                bias_correction2 = 1 - beta2**state['step'].item()
                step_size = group['lr'] *                     math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


# In[16]:


#AGENT! RUNS ENV, 1 step of train and testing as well

import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.done = False
        self.info = None
        self.reward = 0
        self.gpu_id = -1

    def action_train(self):
        if(self.done):
            raise ValueError('Training bug: Agent trying to step while done.')
        
        logit, value = self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))
        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = state
        
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            logit, value = self.model(Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        action = prob.multinomial(1).data
        state, self.reward, self.done, self.info = self.env.step(action.cpu().numpy())

        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        return self


# In[17]:


#utils, imported without changes

import numpy as np
import torch
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# https://bcourses.berkeley.edu/files/70573736/download?download_frd=1&verifier=k02eRurYPQYTzd2i9CF104WBUW6Rx0QdsM6z7Hw9

# In[18]:


#train.py, with modifications
#TRAIN FUNCTION!
#from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
#from environment import atari_env
#from utils import ensure_shared_grads
#from model import teacherNetwork
#from player_util import Agent
from torch.autograd import Variable
import time

numIterSoFar = 0
numIterToNoEntropyLoss = 3000

#def train(rank, args, shared_model, optimizer, env_conf):
def train(env, rank, gamma, tau, model, optimizer, num_iter, setHardIter, num_steps=1000):
    global numIterSoFar
    global numIterToNoEntropyLoss
    log_dir = 'models/'

    if(rank>-1):
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)

    lossLog = []

    state = env.reset()
    player = Agent(None, env, None, state)
    player.model = teacherNetwork(grid_depth, act_dim)

    if(rank>-1):
        player.model.cuda()

    numDone = 0
    totalDone = 0
    ep_lens = []
    print("Process " + str(rank) + " starting training" + ", time is ", end=" ")
    print(datetime.datetime.now().strftime('%H-%M-%S'))
    
    num_eps_per_iter = 1
    bestAvgLen = 1000
    bestAvgLen_score = 0
    
    hardSetting = False

    for iter in range(num_iter):
        #if(iter % setHardIter == 0 and iter > 0):
        #    hardSetting = True #make things hard after the 6kth iteration
        #    print("Making it hard from now on!!!!")
        
        annealing = max(0, 1-numIterSoFar/numIterToNoEntropyLoss)
        
        if(iter%numiters_between_reset_total_count_decrements==0 and iter > 0 and env.reset_total_count>1):
            #env.decrement_reset_total_count() if I had time to build it
            env.reset_total_count -= 1

        if(iter%100==0 and iter > 0):
          if(len(ep_lens)>0):
            avgLen = sum(ep_lens)/len(ep_lens)
            if(bestAvgLen>avgLen):
                bestAvgLen = avgLen
                bestAvgLen_iter = int(iter)
                bestAvgLen_score = str(numDone) + " out of " + str(totalDone)
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}{2}.pt'.format(
                        log_dir, "bestmodel_", rank))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Process " + str(rank) + " Training; current iter: " + str(iter) + ", time is ", end=" ")
            print(datetime.datetime.now().strftime('%H-%M-%S'))
            print("Number of times it's done before maxsteps is " + str(numDone) + " out of " + str(totalDone))
            print("Average episode length is " + str(avgLen))
            print("So far bestAvgLen is " + str(bestAvgLen) + " achieved at iter " + \
                  str(bestAvgLen_iter) + " with " + bestAvgLen_score)
            ep_lens = []
            numDone = 0
            totalDone = 0

        player.model.load_state_dict(model.state_dict())
        
        total_loss = 0
        
        for x in range(num_eps_per_iter):
            for step in range(num_steps):
              player.action_train()
              player.eps_len += 1
              if player.done:
                totalDone += 1
                ep_lens.append(player.eps_len)
                player.eps_len = 0

                if(player.rewards[-1]>0):
                  numDone += 1
                state = player.env.reset(hard = hardSetting)
                player.state = state
                player.done = False #after reset

                break            

            if(rank>-1):
                player.values.append(torch.zeros(1, 1).cuda())
            else:
                player.values.append(torch.zeros(1, 1))

            policy_loss = 0
            value_loss = 0

            if(rank>-1):
                gae = torch.zeros(1, 1).cuda()
            else:
                gae = torch.zeros(1, 1)

            #GAE Calc
            for i in reversed(range(len(player.rewards))):
                R = player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + gamma *                 player.values[i + 1].data - player.values[i].data

                gae = gae * gamma * tau + delta_t

                policy_loss = policy_loss -                 player.log_probs[i] *                 Variable(gae) - annealing * .05 * player.entropies[i]

            #Calculate loss and backprop
            total_loss += policy_loss + 0.5 * value_loss
            lossLog.append(float(policy_loss + 0.5 * value_loss))
            player.clear_actions()

        player.model.zero_grad()
        (total_loss).backward()
        ensure_shared_grads(player.model, model)
        optimizer.step()
        
        numIterSoFar += 1
    
    #once it's done, save the final network
    state_to_save = player.model.state_dict()
    torch.save(state_to_save, '{0}{1}{2}.pt'.format(
        log_dir, "finalmodel_", rank))
    
    return player.model, lossLog


# In[19]:


#test.py, with modifications
#TEST FUNCTION!
#from setproctitle import setproctitle as ptitle
import torch
#from environment import atari_env
#rom utils import setup_logger
#from model import teacherNetwork
#from player_util import Agent
from torch.autograd import Variable
import time
import logging


#def test(args, shared_model, env_conf):
def test(rank, total_num_tests, gamma, tau, input_model, optimizer, env, log_dir):
    log = []
    torch.manual_seed(0)
    
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    state = env.reset()
    player = Agent(input_model, env, None, state) #chaosam fix this line

    player.state = player.env.reset()
    player.eps_len = 0
    
    flag = True
    max_score = 0
    while True:
        player.action_test()
        reward_sum += player.reward
        
        player.eps_len += 1

        if player.done:
            state = player.env.reset()
            
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            
            log.append(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if reward_sum >= max_score: #save best score network
                max_score = reward_sum
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                        log_dir, "bestmodel"))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.state = state
            
            if(num_tests > total_num_tests):
              break

    print("Average performance over " + str(total_num_tests) + " episodes was " + str(reward_mean))


# In[20]:


#TEACHER NETWORK

shared_linear_numhidden = 800 #400
num_outputs = 13
newArchiTry = False #didn't work!

class teacherNetwork(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(teacherNetwork, self).__init__()
        if(not newArchiTry):
            #self.conv0 = nn.Conv2d(num_inputs, num_inputs*4, 3)
            self.conv1 = nn.Conv2d(num_inputs, 256, 2)
            self.conv2 = nn.Conv2d(256, 128, 1)

            self.shared_linear1 = nn.Linear(3200, 1600)
            self.shared_linear2 = nn.Linear(1600, shared_linear_numhidden)

            self.actor_linear = nn.Linear(shared_linear_numhidden, num_outputs)
            self.critic_linear = nn.Linear(shared_linear_numhidden, 1)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 4, 1)
            self.conv2 = nn.Conv2d(4, 4, 1)
            
            maze_size_now = 4*grid_n*grid_n
            maze_size_half = 2*grid_n*grid_n
            just_grid = grid_n*grid_n
            
            self.shared_linear1 = nn.Linear(maze_size_now,maze_size_now)
            self.shared_linear2 = nn.Linear(maze_size_now,maze_size_half)
            self.shared_linear3 = nn.Linear(maze_size_half,just_grid)
            
            self.actor_linear = nn.Linear(just_grid, num_outputs)
            self.critic_linear = nn.Linear(just_grid, 1)
        
        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs
        
        if(not newArchiTry):
            relu = True
            if(relu):
                #x = F.relu(self.conv0(x))
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = F.relu(self.shared_linear1(x))
                x = F.relu(self.shared_linear2(x))
            else:
                x = F.relu(self.conv1(inputs))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = F.sigmoid(self.shared_linear1(x))
                x = F.sigmoid(self.shared_linear2(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(x)
            x = F.relu(self.shared_linear1(x))
            x = F.relu(self.shared_linear2(x))
            x = F.relu(self.shared_linear3(x))

        return self.actor_linear(x), self.critic_linear(x)
# In[21]:


#test forward loop to make sure it runs!
#commented out for now
'''
model = teacherNetwork(10,13)
#model.cuda()
env = MazeWorld(grid_n,blockSide,numCellTypes, \
      ["Picked up greenguy"], \
      100)

state = env.reset()
logit, value = model(state.unsqueeze(0))
#'''


# In[23]:


#MAKE AND PLOT ENV
if(plotstuff):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
import cv2

makeenv = True
init_reset_total_count = 1
numiters_between_reset_total_count_decrements = 500

if(makeenv):
    env = MazeWorld(grid_n,blockSide,numCellTypes, ["Transformed tv"], episode_max_length, \
                    init_reset_total_count, gpu=gpu, changegrid=True)

env.reset()
env.render("step0", 0, "none", 0, "Start", 0)

if(plotstuff):
    img=cv2.imread('step0.jpg')
    plt.axis("off")
    imgplot = plt.imshow(img)


# In[24]:

#houhou
#initialize model and optimizer
loading = False
loadedModelPath = 'models/bestmodel_9.pt'
if(__name__  ==  '__main__'):
    #print(__name__)
    #print("doing this")
    myLr = 0.0001
    shared_model = teacherNetwork(grid_depth, act_dim)
    if(loading):
        bestDict = torch.load(loadedModelPath)
        shared_model.load_state_dict(bestDict)
    shared_model.share_memory()
    if(gpu):
        shared_model.cuda()
    optimizer = SharedRMSprop(shared_model.parameters(), lr=myLr)

# In[25]:


import multiprocessing
#print(multiprocessing.cpu_count())
#print("got to line ~1500")

#torch.cuda.device_count()


# In[26]:


#MAIN TRAINING LOOP (main.py, modified)

import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
#from environment import atari_env # no need we're not doing atari
#from utils import read_config
#from model import teacherNetwork # no need we have a model
#from train import train #see cell before
#from test import test #see cell before
#from shared_optim import SharedRMSprop, SharedAdam #also imported
#from gym.configuration import undo_logger_setup
import time

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

#torch.manual_seed(0)
total_num_tests = 50
gamma = 0.99
tau = 0.96

loss_logs = []
rank  = 1

parallel = True
numWorkers = 10
torch.manual_seed(0)

if(parallel and wantToTrain):
    if(__name__  ==  '__main__'):

        if(gpu):
            torch.cuda.manual_seed(0)
            #mp.set_start_method('spawn')#, force=True)
            print()

        processes = []

        #numWorkers = 8

        for rank in range(0, numWorkers):
            print(rank)
            if(not gpu):
                rank = -1
            p = mp.Process(
                target=train, args=(env, rank,  gamma, tau, shared_model, optimizer,12001, 7000))#16001
            p.start()
            processes.append(p)
            time.sleep(0.1)

        for p in processes:
            time.sleep(0.1)
            p.join()
elif(wantToTrain): #yes I know this is horrible
    test(rank, total_num_tests, gamma, tau, shared_model, optimizer, env, 'sam')
    model, loss_log = train(env, rank,  gamma, tau, shared_model, optimizer,500)
    loss_logs = loss_logs + loss_log
    test(rank, total_num_tests, gamma, tau, shared_model, optimizer, env, 'sam')

# for param in model.parameters():
#   print(param.data)
#test(rank, gpu_ids, num_steps, gamma, tau, shared_model, optimizer, env, log_dir):
#backtobottomofmain

if(wantToTrain and __name__  ==  '__main__'):
    #samget
    log_dir = 'models/'
    state_to_save = shared_model.state_dict()
    torch.save(state_to_save, '{0}{1}.pt'.format(log_dir, "finalModel"))

# In[27]:


#RUN n EPISODEs AND SAVE IT WITH NETWORK
#samback

if(__name__  ==  '__main__' and wantToTest):
    #print("hello, world")
    run_n_episodes = True
    n = 100
    avgNumSteps = []
    printingresults = False
    makeimages = True #make an image for each step
    justprintsummary = True #only print numsteps and cumreward @ the end
    makegif = True

    if(run_n_episodes):
        numSolved = 0
        for x in range(n):
            
            state = env.reset()
            submode = mode+"_" + str(x) + "_" + str(rank) + "_"

            if(makeimages or makegif):
              for fname in os.listdir('./'):
                if fname.startswith(submode) and fname.endswith(".jpg"):
                    softremove(os.path.join('.', fname))
              env.render(submode+ "0", 0, "none", 0, "Start", 0)
            if(logging):
              envlog = torch.zeros((episode_max_length+3,env.numCellTypes,env.grid_n,env.grid_n))#,\
               #dtype=torch.int32)
              envlog[0] = env.t
              actRewDoneLog = []


            model = shared_model
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #                          Run an episode
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            done = False
            i = 1
            cumreward = 0

            while not done:
              logit, value = model(state.unsqueeze(0))
              prob = F.softmax(logit, dim=1)
              action = prob.multinomial(1).data 

              state, reward, done, taskupdate = env.step(int(action[0]))
              renderfilename = submode + str(i)

              if(logging):
                envlog[i] = env.t
                actRewDoneLog.append([action, reward, done])

              actText = actionNumToText(int(action[0]))

              if(printingresults):
                print(str(i) + ": Action was " + actText +       ", reward is " + str(reward) + ", taskupdate is "       + taskupdate + ".")

              if(makeimages or makegif):
                env.render(renderfilename, i, int(action[0]), reward, taskupdate, cumreward)

              i += 1
              cumreward += reward

            if(reward>0):
                numSolved += 1
            else:
                print("Episode " + str(x) + "didn't solve. See " + submode + ".gif for episode") 
                
            if(printingresults or justprintsummary):
              print("Ep " + str(x) + ": Env terminated after " + str(i-1) + " steps, cumreward = " + str(cumreward))
            
            avgNumSteps.append(i-1)

            if(logging):
              softremove('envlog.tensor')
              torch.save(envlog, 'envlog.tensor')

            if(makegif):
              #code from Alex Buzunov at https://goo.gl/g2G9c6
              duration = 1
              filenames = sorted(filter(os.path.isfile, [x for x in os.listdir()   if (x.endswith(".jpg") and x.startswith(submode))]), key=lambda   p: os.path.exists(p) and os.stat(p).st_mtime or   time.mktime(datetime.now().timetuple()))

              #make an image that says "START"
              result = env.makeTextImage("START")
              result.save(submode + "Start.jpg")

              #make an image that says "END"
              result = env.makeTextImage("END")
              result.save(submode + "End.jpg")
              filenames.append(submode+"End.jpg")

              images = []
              images.append(imageio.imread(submode + "Start.jpg"))
              for filename in filenames:
                images.append(imageio.imread(filename))

              #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
              output_file = submode + '.gif'
              imageio.mimsave(output_file, images, duration=duration)

            #EXPORT EPISODE TO DRIVE
            uploadIndividualImages = False #otherwise just upload gif
            if(uploadIndividualImages and goog):
              for filename in filenames:
                f = drive.CreateFile()
                f.SetContentFile(filename) #add the contents of your local file to the remove file
                f.Upload() #create your remote file

            if(goog):
                f = drive.CreateFile()
                f.SetContentFile(mode+'.gif')
                f.Upload()
            
            #get rid of all the jpgs that start with submode
            if(makeimages or makegif):
              for fname in os.listdir('./'):
                if fname.startswith(submode) and fname.endswith(".jpg"):
                    softremove(os.path.join('.', fname))
        
        print("Thanks for playing! numSolved is " + str(numSolved) + " out of " + str(n))
        print("Average number of steps is " + str(sum(avgNumSteps)/len(avgNumSteps)))