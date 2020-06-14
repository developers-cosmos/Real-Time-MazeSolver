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

# splitting the channels of maze
image = cv2.imread('test1.png')
image = cv2.resize(image,(255,255))
b, g, red = cv2.split(image)
h, w = b.shape

# convert the maze image to grayscale
image = Image.open('test1.png').convert('L')

print(image.size)
image = image.resize((w,h))
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
b = np.pad(b, pad_width=5,mode='constant',constant_values=0)
g = np.pad(g, pad_width=5,mode='constant',constant_values=0)
red = np.pad(red, pad_width=5,mode='constant',constant_values=0)


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


#================================ Finding the shortest path to reach the Stop from Start ====================================

def AStar(start, goal, neighbor_nodes, distance, cost_estimate):
    def reconstruct_path(came_from, current_node):
        path = []
        while current_node is not None:
            path.append(current_node)
            current_node = came_from[current_node]
        return list(reversed(path))

    g_score = {start: 0}
    f_score = {start: g_score[start] + cost_estimate(start, goal)}
    openset = {start}
    closedset = set()
    came_from = {start: None}

    while openset:
        current = min(openset, key=lambda x: f_score[x])
        if current == goal:
            return reconstruct_path(came_from, goal)
        openset.remove(current)
        closedset.add(current)
        for neighbor in neighbor_nodes(current):
            if neighbor in closedset:
                continue
            if neighbor not in openset:
                openset.add(neighbor)
            tentative_g_score = g_score[current] + distance(current, neighbor)
            if tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + cost_estimate(neighbor, goal)
    return []


def is_blocked(p):
    x,y = p
    pixel = maze_input[x,y]
    if (pixel == 0):
        return True

def von_neumann_neighbors(p):
    x, y = p
    neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]
    return [p for p in neighbors if not is_blocked(p)]

def manhattan(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def squared_euclidean(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


distance = squared_euclidean
heuristic = squared_euclidean

path = AStar(start, goal, von_neumann_neighbors, distance, heuristic)

for position in path:
    x,y = position
    maze_input[x,y] = 9 # 9 represents the result



# add the neighbours to the path to increase thickness
ke = 5
kernel = np.ones((ke, ke), np.uint8) * 9
maze_input = cv2.dilate(maze_input, kernel, iterations=1)


# set the 0 and 1's to 3D values
for r in range(h):
    for c in range(w):
        if maze_input[r,c] == 1:
            maze_input[r,c] = 255
        elif maze_input[r,c] == 0:
            maze_input[r,c] = 0
        elif maze_input[r,c] == 9:
            maze_input[r,c] = 0


g = cv2.bitwise_and(g, g, mask=maze_input)
b = cv2.bitwise_and(b, b, mask=maze_input)


# merge the channles of maze
res = cv2.merge((b, g, red))
res = res[5:-6, 5:-6] # remove the borders

cv2.imwrite('SolvedMaze.jpg', res)
res = cv2.imread('SolvedMaze.jpg')
res = cv2.resize(res,(500,500))
cv2.imshow('Solved Maze', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
