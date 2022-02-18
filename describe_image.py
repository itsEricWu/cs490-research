from os import listdir
import os
from os.path import isfile, join
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle


def select_one_file_from_each_folder():
    selected_files = []
    for i in range(19):
        mypath = f"Test/{i}"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        selected_file = random.choice(onlyfiles)
        selected_files.append([i, selected_file])
    return selected_files

def plot_gray_histogram(input_file,output_path, i):
    img = cv2.imread(input_file, 0)
    ravel_array = img.ravel() # 1d array
    mean = np.mean(ravel_array)
    variance = np.var(ravel_array)
    plt.clf() 
    #plt.hist(ravel_array,256,[0,256])
    #plt.title(f"grey_histogram{i}")
    return plt, mean, variance
def plot_rgb_histogram(input_file,output_path, i):
    img = cv2.imread(input_file)
    color = ('b','g','r')
    img_ravel = [img[:,:,0].ravel(), img[:,:,1].ravel(),img[:,:,2].ravel()] #bgr
    means = [np.mean(img_ravel[0]), np.mean(img_ravel[1]), np.mean(img_ravel[2])] #bgr
    variances = [np.var(img_ravel[0]), np.var(img_ravel[1]), np.var(img_ravel[2])] #bgr
    #plt.hist(img_ravel,color = color, label = color)
    plt.clf()
    #for k,col in enumerate(color):
    #    histr = cv2.calcHist([img], [k], None, [256], [0,256])
    #    plt.plot(histr,color = col)
    #    plt.xlim([0,256])
    #plt.title(f"color_histogram{i}")
    return plt, means, variances


def plot_one_file_from_each_folder():
    #selected_files = select_one_file_from_each_folder()
    #print(selected_files)
    selected_files = [[0, '1154.ppm'], [1, '973.ppm'], [2, '864.ppm'], [3, '488.ppm'], [4, '1767.ppm'], [5, '7.ppm'], [6, '407.ppm'], [7, '253.ppm'], [8, '65.ppm'], [9, '877.ppm'], [10, '892.ppm'], [11, '294.ppm'], [12, '148.ppm'], [13, '87.ppm'], [14, '196.ppm'], [15, '547.ppm'], [16, '174.ppm'], [17, '105.ppm'], [18, '371.ppm']]
    output_path = 'pixel_histograms'
    for i in range(len(selected_files)): # i is original image folder name
        input_file = f"Test/{selected_files[i][0]}/{selected_files[i][1]}"
        plt_grey, grey_mean, grey_variance = plot_gray_histogram(input_file,output_path, i) 
        plt_grey.savefig(f'{output_path}/grey_histogram{i}.jpg')
        plt_color, means, variances = plot_rgb_histogram(input_file,output_path, i)
        plt_color.savefig(f'{output_path}/color_histogram{i}.jpg')
        
    return selected_files



def main():   
    #selected_files = plot_one_file_from_each_folder()
    # initilize dataframe for mean and variance
    column_name = ['folder_name', 'image_name', 'grey_mean', 'grey_variance', 'b_mean', 'b_variance', 'g_mean', 'g_variance', 'r_mean', 'r_variance']
    df = pd.DataFrame(columns = column_name)
    # read all files
    parent_dir = "pixel_histograms"
    for i in range(19):
        input_path = f"Test/{i}"
        output_path = os.path.join(parent_dir, str(i))
        isExist = os.path.exists(output_path)
        if not isExist:
            os.mkdir(output_path)
            print("Directory '%s' created" %output_path)
        files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
        for j in range(len(files)):
            input_file = f"Test/{i}/{files[j]}"
            plt_grey, grey_mean, grey_variance = plot_gray_histogram(input_file,output_path, j) 
            #plt_grey.savefig(f'{output_path}/grey_histogram{j}.jpg')
            plt_color, means, variances = plot_rgb_histogram(input_file,output_path, j)
            #plt_color.savefig(f'{output_path}/color_histogram{j}.jpg')
            #row = [folder_name, image_name, grey_mean, grey_variance, b_mean, b_variance, g_mean, g_variance, r_mean, r_variance]
            row_df = {"folder_name" : str(i),
                        "image_name" : str(j),
                        "grey_mean" : grey_mean,
                        "grey_variance" : grey_variance, 
                        "b_mean" : means[0], 
                        "b_variance" : variances[0], 
                        "g_mean" : means[1], 
                        "g_variance" : variances[1], 
                        "r_mean" : means[2], 
                        "r_variance" : variances[2]}
            df = df.append(row_df, ignore_index = True)
    save_path = "generated/image_description_results"
    isExist = os.path.exists(save_path)
    if not isExist:
        os.mkdir(save_path)
        print("Directory '%s' created" %save_path)
    pickle.dump(df, open(save_path+"/pixel_mean_variance", "wb"))



        
    
    
main()
    

