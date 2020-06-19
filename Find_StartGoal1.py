#######################################################################################
###################         authors = "Saiteja Kura, RitheeshBaradwaj" ################
###################         project name = "Real Time Maze Solver"     ################
###################         credits = "Developers Cosmos"              ################
###################         license = "Apache License 2.0"             ################
###################         version = "1.0.0"                          ################
###################         maintainer = "Saiteja Kura, RitheeshBaradwaj" #############
###################         email = "developerscosmos6@gmail.com"      ################
###################         status = "Production"                      ################
#######################################################################################
#######################################################################################

# import the required packages
from PIL import Image
import sys
import numpy as np
import cv2

# convert the maze image to grayscale
image = Image.open('test5.png').convert('L')
w, h = image.size # get the dimenssions

# set all white pixels to 1 and black to 0
binary = image.point(lambda p: p > 128 and True)  # 0 is blocked and 1 is unblocked

# Resize the binary so that we can reduce lot of 0's and 1's to traverse - this value is hardcoded for now
# we need to figure out a way to set w,h value without lossing information
# binary = binary.resize((500,800),Image.NEAREST)
w, h = binary.size

# converting image to numpy array
maze_input = np.array(binary)


#==================================== Detecting Start and Stop Points for the Maze ===================================

# Padding for neighbors
maze_input = np.pad(maze_input, pad_width=5, mode='constant', constant_values=0)

# To identify which has the opening and closing
points = [maze_input[5],maze_input[h+4],maze_input[:,5],maze_input[:,w+4]]
# points = [maze_input[0],maze_input[h-1],maze_input[:,0],maze_input[:,w-1]]
res=[]
point = 0

for i in range(len(points)):
    count=0
    if(len(res)<2):
        point= np.sum(points[i])
        if(point!=0):
            if(i<=1):
                for j in range(len(points[i])):
                    if(i==0):
                        k=5
                    if(i==1):
                        k=h+4
                    if(maze_input[k][j]==0):
                        continue
                    if(maze_input[k][j]==1):
                        count+=1
                        if(count==int(point/2)):
                            res.append((k,j))
                            break
            else:
                for j in range(len(points[i])):
                    if(i==2):
                        k=5
                    if(i==3):
                        k=w+4
                    if(maze_input[j][k]==0):
                        continue
                    if(maze_input[j][k]==1):
                        count+=1
                        if(count==point/2):
                            res.append((j,k))

    else:
        break

# Start and Stop Points
start=res[0]
goal=res[1]