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

# ===========================================================================================================================================

camera_port=0
cap = cv2.VideoCapture('Maze1.mp4')
img_array = []
while True:
    ret,img=cap.read()
    try:
        if not img:
            print("no im")
            break
        else:
            cv2.imwrite('input.png',img)
    except:
        cv2.imwrite('input.png',img)
        pass
    img = cv2.imread('input.png',cv2.IMREAD_GRAYSCALE)
    final = img
    # =======================================================
    try:

        processed = pre_process_image(img)
        corners = find_corners_of_largest_polygon(processed)
        display_points(processed, corners)
        corners = np.array(corners)
        crop = cv2.imread('crop.png')

        cropped = crop_and_warp(crop, corners)
        cv2.imwrite('cropped.png',cropped)
        # cv2.imshow('cropped image', cropped)
        # cv2.waitKey(0)

        cropped = cv2.imread('cropped.png')
        cropped = cropped[5:-5,5:-5]

        cv2.imwrite('padding_crop.png',cropped)

        cropped = Image.open('padding_crop.png').convert('L')
        w1,h1 = cropped.size
        cropped = cropped.resize((w1+500,h1+500))
        cropped = cropped.point(lambda p: p > 128 and True)

        # cv2.imshow('cropped image padding', cropped)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        crop_width, crop_height = cropped.size
        p1 = 0

        for r in range(crop_width):
            if p1 != 0:
                break
            for c in range(crop_height):
                o = r,c
                val = cropped.getpixel(o)
                # print(val)
                if val == 1:
                    p1, p2 = r, c
                    print(val,p1,p2)
                    break
        cropped = cv2.imread('cropped.png')
        cropped = cropped[12:-12,12:-12]
        maze = cropped
        # cv2.imshow('Maze Input', maze)
        cv2.imwrite('maze.png',maze)
        # cv2.waitKey(0)
        # maze = cropped[p1-10:p1+520, p2-10:p2+510]
        # print(maze.shape)

        for r in range(w1-24):
            for c in range(h1-24):
                if sum(maze[r,c]) == 765:
                    maze[r,c] = (0,0,0)
                else:
                    maze[r,c] = (255,255,255)


        maze = cropped[3:-10,2:-7]
        # cv2.imshow('Maze Input', maze)
        cv2.imwrite('maze.png',maze)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #====================================================================================================================

        from PIL import Image
        import sys
        import numpy as np
        import cv2

        image = cv2.imread('maze.png')
        image = cv2.resize(image,(255,255))
        b, g, red = cv2.split(image)

        h, w = b.shape
        image = Image.open('maze.png').convert('L')

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
        # print(b.shape,g.shape,r.shape)

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

        print(start,goal)
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


        result = np.zeros([h+4,w+4,3]) # an empty image

        # set the 0 and 1's to 3D values
        for r in range(h):
            for c in range(w):
                if maze_input[r,c] == 1:
                    
                    result[r,c,:] = (255,255,255)
                    maze_input[r,c] = 255
                elif maze_input[r,c] == 0:
                    result[r,c,:] = (0,0,0)
                    maze_input[r,c] = 0
                elif maze_input[r,c] == 9:
                    result[r,c,:] = (0,0,0)
                    maze_input[r,c] = 0


        print(g.shape,b.shape,red.shape,maze_input.shape)

        g = cv2.bitwise_and(g, g, mask=maze_input)
        b = cv2.bitwise_and(b, b, mask=maze_input)



        res = cv2.merge((b, g, red))
        res = res[5:-6, 5:-6]
        cv2.imwrite('SolvedMaze.jpg', res)
        res = cv2.imread('SolvedMaze.jpg')
        res = cv2.resize(res,(460,500))
        
        # cv2.waitKey(0)
        crop = cv2.imread('crop.png')
        crop = cv2.resize(crop,(270,500))
        
        final = cv2.imread("input.png")
        final = cv2.resize(final,(270,500))
        print("shape",res.shape,final.shape,crop.shape)
        stacked = np.hstack((final,crop,res))

        # cv2.imshow('Real Time Maze Solver',stacked)
        img_array.append(stacked)
    except:
        pass
    
    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
size = (1000,500)
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()