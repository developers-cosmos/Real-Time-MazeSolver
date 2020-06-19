def start_goal():
    try:
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
    except Exception as e:
        print(e)
