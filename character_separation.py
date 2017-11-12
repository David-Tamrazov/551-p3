import numpy as np
import pandas as pd
import cv2 
import os
import errno

from matplotlib import pyplot as plt
from scipy import misc

resize = misc.imresize

def fetch_data(development=True):
    
    if (development):
        # Testing code
        df_x = pd.read_csv("train_x.csv", nrows=100)
        df_y = pd.read_csv("train_y.csv", nrows=100)

        X_train = df_x.values
        X_test = ""
        Y_train = df_y.values
    
    else:
        # Production code - uncomment for submission
        x = np.loadtxt("train_x.csv", delimiter=",")
        y = np.loadtxt("train_y.csv", delimiter=",")
        x_t = np.loadtxt("test_x.csv", delimiter=",")

        X = x
        X_test = x_t
        Y_train = y
    
    return X_train, Y_train, X_test     
    

def show_images(imgs):
    
    for img in imgs:
        print img.shape
        implot = plt.imshow(img)
        plt.show()


def preprocess_images(X):
    
    X_preprocessed = []
    
    for idx, image in enumerate(X):
        
        # reshape the image and retype it to uint8 
        reshaped_img = np.reshape(image, (64, 64)).astype(np.uint8)
        
        # enlarge the image to cleanly separate into three 28x28 images later on  
        resized_img = resize(reshaped_img, (128, 128))
        
        # remove the noise from the image
        clean_img = cv2.fastNlMeansDenoising(resized_img, None, 30, 7, 21)
        
        # convert to binary 
        _, bin_img = cv2.threshold(clean_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        X_preprocessed.append(bin_img)
    
    return X_preprocessed


def extract_characters(img): 

    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    
    ctr = 0
    
    for contour in contours:
        ctr += 1
    
    # it found 3 characters in the image - extract the characters using their contours
    if (ctr == 3): return extract_contours(img)
    
    # characters are either merged or there's too much noise in the image- use kmeans to separate the data 
    else: return extract_kmeans_clusters(img)
    

def extract_kmeans_clusters(img):
    
    # black matrices to project the images on to later
    fst = np.zeros((img.shape))
    scd = np.zeros((img.shape))
    trd = np.zeros((img.shape))

    # empty list of points to pass to KMeans
    pts = []


    for (x, y), pxl in np.ndenumerate(img):

        # if this is a white pixel, add it to the list of points 
        if (pxl != 0):
            pts.append([x,y])

    # convert to np.float32
    X = np.float32(pts)

    # define criteria for kmeans - termination condition, nu iterations, epsilon for accuracy tolernace
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)

    # apply kmeans and get the label for all points 
    _, labels, _ = cv2.kmeans(X, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # enumerate over the labels and project the separated points onto the black matrices
    for idx, label in enumerate(labels):

        # set the pixel as white on the first black matrix
        if (label[0] == 0):
            coords = X[idx]
            x = int(coords[0])
            y = int(coords[1])
            fst[x, y] = 255

        # set the pixel as white on the second black matrix
        elif (label[0] == 1):
            coords = X[idx]
            x = int(coords[0])
            y = int(coords[1])
            scd[x, y] = 255

        # set the pixel as white on the third black matrix
        else:
            coords = X[idx]
            x = int(coords[0])
            y = int(coords[1])
            trd[x, y] = 255   
    
    
    # cleanly scale the image back to 28x28 through its contours
    fst_scaled = extract_contours(fst.astype(np.uint8), True)[0]
    scd_scaled = extract_contours(scd.astype(np.uint8), True)[0]
    trd_scaled = extract_contours(trd.astype(np.uint8), True)[0]

    return [fst_scaled, scd_scaled, trd_scaled]
   

def extract_contours(img, single=False):
    
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        
    bounding_boxes = get_bounding_boxes(contours, single)
    
    char_imgs = []
    
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        
        # get the image within this bounding box
        char_image = img[y:y+h,x:x+w]
        
        # convert it back to binary (resize converts to grayscale)
        im = resize(char_image, (28, 28)).astype(np.uint8)
        _, bin_img = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # append  to images
        char_imgs.append(bin_img)
        
    return char_imgs



def get_bounding_boxes(contours, single=False):
    
    bounding_boxes = []
    
    # if we're trying to get a single bounding box
    if (single):
        
        # combine the contours into one contour 
        ctrs = []
        
        for contour in contours:
            
            area = cv2.contourArea(contour, False);                                                
            
            if(area > 20):                                                                                                                      
                ctrs.append(contour)
              
            
        merged_contours = np.concatenate(ctrs, axis=0)
        
        # create a bounding box out of the one contour
        x, y, w, h = cv2.boundingRect(np.asarray(merged_contours))
        bbox = (x, y, w, h)
        
        bounding_boxes.append(bbox)
       
    # we're trying to find multiple characters
    else :
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # max constraints to keep box bounded within the scope of the image
            bbox = (max(x-2, 0), max(y-2, 0), max(w+4, 128), max(h+4, 128))
            bounding_boxes.append(bbox)
        
    return bounding_boxes

def separate_images(X):
    
    # empty list to store groups of 3 images 
    img_list = []
    
    # create a list that stores 
    for img in X:
        
        # extract the characters in the image as 3 separate images in a list
        char_imgs = extract_characters(img)
        
        # shape the image back to 1 x D for writing to file 
        char_imgs = [shape_to_one(img) for img in char_imgs]

        # append the list to the group of lists 
        img_list.append(char_imgs)
    
    return img_list 

def shape_to_one(img):
    
    # reshape the image back to one line  
    dims = img.shape[0] * img.shape[1]
    return np.reshape(img, (1, dims))


def write_to_file(X, img_dir):
    
    def write_list_to_file(lst, dir):
        
        for idx, img in enumerate(lst): 

            pathname = img_dir + str(idx)

            print pathname
            
            np.savetxt(pathname, img, delimiter=',')
        

    for idx, img_list in enumerate(X):
    
        pathname = img_dir + str(idx)

        # write the image list to file 
        write_list_to_file(img_list, pathname)


def main():
    
    X_train, Y_train, X_test = fetch_data()

    print "Preprocessing the images..."

    # denoise the data and convert to binary
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    print "Separating images..."

    X_train = separate_images(X_train)
    X_test = separate_images(X_test)

   
    print "Writing images to file..."

    write_to_file(X_train, 'train/')
    write_to_file(X_test, 'test/')

    print "Done."
        

main()