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

from PIL import Image
import sys
import numpy as np
import cv2
import operator

def pre_process_image(img, skip_dilate=False):
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

def find_corners_of_largest_polygon(img):
	"""Finds the 4 extreme corners of the largest contour in the image."""
	contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
	contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
	polygon = contours[0]  # Largest image
	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	# Return an array of all 4 points using the indices
	# Each point is in its own array of one coordinate
	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def show_image(img,points):
	"""Shows an image until any key is pressed"""
	cv2.imwrite('crop.png',img)
	
    #cv2.imshow('image', img)  # Display the image
	# cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
	# cv2.destroyAllWindows()  # Close all windows

def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
	img = in_img.copy()

	# Dynamically change to a colour image if necessary
	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
            
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	for point in points:
		img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
	show_image(img,points)
	return img

def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	# Rectangle described by top left, top right, bottom right and bottom left points
	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

	# Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	# Get the longest side in the rectangle
	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	# Describe a square with side of the calculated length, this is the new perspective we want to warp to
	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

	# Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
	m = cv2.getPerspectiveTransform(src, dst)

	# Performs the transformation on the original image
	return cv2.warpPerspective(img, m, (int(side), int(side)))

def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	show_image(np.concatenate(rows))

camera_port=0
cap = cv2.VideoCapture('Maze2.mp4')
img_array = []
key = 1
while True:
    ret,img=cap.read()
    try:
        if not img:
            print("no im")
            break
        else:
            img = cv2.resize(img,(1000,500))
            cv2.imwrite('input.png',img)
    except:
        img = cv2.resize(img,(1000,500))
        cv2.imwrite('input.png',img)
        pass
    img = cv2.imread('input.png',cv2.IMREAD_GRAYSCALE)
    final = img
    try:
        img = cv2.imread('input.png',cv2.IMREAD_GRAYSCALE)
        print("image shape",img.shape)
        processed = pre_process_image(img)
        corners = find_corners_of_largest_polygon(processed)
        display_points(processed, corners)
        corners = np.array(corners)
        print(corners)

        # ======================================================================================================================================
        im = np.array(Image.open('input.png').convert('L'))

        # Get height and width
        h,w = im.shape[0:2]

        # Make a single pixel wide column, same height as image to store row sums in
        rowsums=np.empty((h))      
        # Sum all pixels in each row
        np.sum(im,axis=1,out=rowsums)        
        # Normalize to range 0..100, if rowsum[i] < 30 that means fewer than 30% of the pixels in row i are white
        rowsums /= np.max(rowsums)/100      

        # Find first and last row that is largely black
        first = last = -1
        for r in range(corners[0][1] + 23,corners[2][1]-21):
            if first < 0 and rowsums[r] < 100:
                first = r
            if rowsums[r] < 100:
                last = r
        

        print(first,"first")
        print(last,"last")
        colsums=np.empty((w))      
        # Sum all pixels in each col
        np.sum(im,axis=0,out=colsums)        
        # Normalize to range 0..100, if colsum[i] < 30 that means fewer than 30% of the pixels in col i are white
        colsums /= np.max(colsums)/100      

        # Find first and last col that is largely black
        first = last = -1

        for c in range(corners[0][0]+20,corners[2][0]-20):
            if first < 0 and colsums[c] < 100:
                first = c
            if colsums[c] < 100:
                last = c
        

        print(first,"first")
        print(last,"last")

        # cv2.circle(img,(120,445),radius=5,color=(0,255,0))
        # cv2.circle(img,(595,899),radius=5,color=(0,255,0))
        img = cv2.resize(img,(600,600))
        # cv2.circle(im,(120,440),radius=5,color=(0,255,0))
        # cv2.circle(im,last,radius=5,color=(0,255,0))

        p = True
        img = cv2.imread('input.png')
        print("image shape",img.shape)
        for r in range(corners[0][1] + 23,corners[2][1]-21):
            try:
                if not p:
                    break
                for c in range(corners[0][0]+20,corners[2][0]-20):
                    if sum(img[r,c]) < 600:
                        start = (r,c+5)
                        p = False
                        break
                
            except Exception as e:
                print(e)
        # print(start)
        p = True # 
        for r in range(corners[2][1]-21, corners[0][1] + 23,-1):
            if not p:
                break
            for c in range(corners[2][0]-20,corners[0][0]+20,-1):
                if sum(img[r,c]) < 600:
                    stop = (r,c+5)
                    p = False
                    break

        print("start",start)
        print("stop",stop)

        cv2.circle(img,(start[1]+50,start[0]-10),radius=5,color=(255,0,0))
        cv2.circle(img,(stop[1]-100,stop[0]),radius=5,color=(255,0,0))
        # cv2.imshow('',img)
        # cv2.waitKey(0)
        img = cv2.resize(img,(600,600))
        # cv2.circle(im,(120,440),radius=5,color=(0,255,0))
        # cv2.circle(im,last,radius=5,color=(0,255,0))
        

        start = (start[0]-10,start[1]+50)
        goal = (stop[0],stop[1]-100)

        # ==========================================================================================================================================

        image = Image.open('input.png').convert('L')
        w, h = image.size # get the dimenssions

        # set all white pixels to 1 and black to 0
        binary = image.point(lambda p: p > 147 and True)  # 0 is blocked and 1 is unblocked

        # Resize the binary so that we can reduce lot of 0's and 1's to traverse - this value is hardcoded for now
        # we need to figure out a way to set w,h value without lossing information
        # binary = binary.resize((500,800),Image.NEAREST)
        w, h = binary.size

        # converting image to numpy array
        maze_input = np.array(binary)


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
            maze_input[x,y] = 9
            maze_input[x,y-1] = 9
            maze_input[x,y+1] = 9
            maze_input[x-1,y] = 9
            maze_input[x+1,y] = 9

        '''for r in range(h):
            for c in range(w):
                i=1
                j=1
                if maze_input[r,c] == 9:
                    while(maze_input[r,c+i] == 1 and i<=5 and c+i<w):
                        maze_input[r,c+i] = 2 # 2 is the neighbour of 9 which is unblocked
                        i=i+1
                    while(maze_input[r,c-i] == 1 and 10-i>=0 and c-i>w):
                        maze_input[r,c-i] = 2
                        i=i+1
                    while(maze_input[r+j,c] == 1 and 10-j>=0 and c-j>h):
                        maze_input[r+j,c] = 2
                        j=j+1
                    while(maze_input[r-j,c] == 1 and 10-j>=0 and c-j>h):
                        maze_input[r-j,c] = 2
                        j=j+1'''

        img = cv2.imread('input.png')
        print("image shape",img.shape)
        import time
        for r in range(h):
            for c in range(w):
                if maze_input[r,c] == 2 or maze_input[r,c] == 9:
                    img[r,c,:] = (0,0,255)
        cv2.imwrite("result.jpg",img)
        img = cv2.imread("result.jpg")
        # img = cv2.resize(result,(700,700))
        
    except Exception as e:
        print(e)
    
    img_array.append(img)
    # cv2.imshow("result",img)
    # key = cv2.waitKey(1)
    
    if key == -3:
        break
    key +=1

cap.release()
cv2.destroyAllWindows()
size = (1000,500)
out = cv2.VideoWriter('maze2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
