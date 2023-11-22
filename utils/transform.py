import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
EXAMPLE_IMAGE_GREYSCALE = f"{ROOT_DIR}/utils/squareX.bmp"
EXAMPLE_IMAGE_COLOR_SQUARE = f"{ROOT_DIR}/utils/00260_0.jpg"
EXAMPLE_IMAGE_COLOR_OBLONG = f"{ROOT_DIR}/utils/00216_1.jpg"


def TwoDHaarTransform(Im, L):
    # This program was written to implement the Haar Wavelet on an image.  
    #
    # Inputs:
    #       Im is the image to be transformed.
    #       L is how many levels of decomposition are to be executed.
    #
    # Output:
    #       waveletIm is the transformed image coefficients from the subbands 
    #       {LL, LH, HL, HH} for each level

    waveletIm = Im.copy()
    
    N1, N2 = Im.shape
    waveletIm = np.zeros((N1, N2))  # Create an array with zeros for waveletIm
    for l in range(L):
        n1 = N1 // 2**(l)
        n2 = N2 // 2**(l)
        LL = np.zeros((n1 // 2, n2 // 2), dtype=float)
        LH = np.zeros((n1 // 2, n2 // 2), dtype=float)
        HL = np.zeros((n1 // 2, n2 // 2), dtype=float)
        HH = np.zeros((n1 // 2, n2 // 2), dtype=float)
        
        for i in range(0, n1, 2):
            for j in range(0, n2, 2):
                im00 = Im[i, j]
                im01 = Im[i, j + 1] 
                im10 = Im[i + 1, j]
                im11 = Im[i + 1, j + 1]
                
                LL[(i + 1) // 2, (j + 1) // 2] = (im00 + im01 + im10 + im11) / 4
                LH[(i + 1) // 2, (j + 1) // 2] = (im00 + im01 - im10 - im11) / 4
                HL[(i + 1) // 2, (j + 1) // 2] = (im00 - im01 + im10 - im11) / 4
                HH[(i + 1) // 2, (j + 1) // 2] = (im00 - im01 - im10 + im11) / 4
        
        synImage = np.zeros((n1, n2), dtype=float)
        synImage[0:n1 // 2, 0:n2 // 2] = LL
        synImage[n1 // 2:n1, 0:n2 // 2] = LH
        synImage[0:n1 // 2, n2 // 2:n2] = HL
        synImage[n1 // 2:n1, n2 // 2:n2] = HH

        Im = LL
        waveletIm[0:n1, 0:n2] = synImage

    return waveletIm


def plotTwoDHaar(filepath, image_dim = 128):
    # Display 2D Haar Wavelet Transform (of greyscale image).

    # Load (and display) the original image
    Im = cv.imread(filepath)
    Im = cv.cvtColor(Im, cv.COLOR_BGR2RGB) # opencv uses bgr color mode and matplotlib uses rgb color mode
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255) # cmap is ignored if Im is RGB(A).
    plt.title(os.path.basename(filepath))
    print("Original image size: ", Im.shape)

    # Resize
    # Other interpolation algorithms: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    # cv.INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results.
    # But when the image is zoomed, it is similar to the cv.INTER_NEAREST method.
    Im_resized = cv.resize(Im, (image_dim, image_dim), interpolation = cv.INTER_AREA) # INTER_LINEAR - bilinear interpolation
    # Convert to greyscale
    Im = cv.cvtColor(Im_resized, cv.COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(f"Resized (& grayscale) version of {os.path.basename(filepath)}")
    print("\nResized (& grayscale) image size: ", image_dim, image_dim)

    # Perform the 2D Haar Wavelet Transform
    L = 4 # Define the number of levels L
    waveletIm = TwoDHaarTransform(Im.astype(float), L)
    result = np.abs(waveletIm)
    
    # Display the transformed image
    plt.figure()
    plt.imshow(result, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(f"2D Haar Wavelet Transform of {os.path.basename(filepath)}")
    print("\nFinal result size: ", result.shape)

    # Show figures
    plt.show()


def NPointDCT2(N):
    # This function produces the N-point DCT_2 of an input size received.
    # The N-point DCT_2 matrix will be a square matrix in this function.
    # This is code that provides the use of the Discrete Cosine Transform 
    # for transforming a spacial domain image to a Fourier domain.
    # 
    # Inputs:
    #       N is the square image dimensions
    # 
    # Output:
    #       NPDCT is N-point DCT_2 of an input size received

    NPDCT = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                NPDCT[k, n] = np.sqrt(1/N)
            else:
                temp = np.cos((np.pi * (2 * n + 1) * k) / (2 * N))
                NPDCT[k, n] = np.sqrt(2 / N) * temp
    return NPDCT


def plotDCT2(filepath, image_dim = 128):
    # Display DCT2 Transform (of greyscale image).

    # Load (and display) the original image
    Im = cv.imread(filepath)
    Im = cv.cvtColor(Im, cv.COLOR_BGR2RGB) # opencv uses bgr color mode and matplotlib uses rgb color mode
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255) # cmap is ignored if Im is RGB(A).
    plt.title(os.path.basename(filepath))
    print("Original image size: ", Im.shape)
    
    # Resize
    # Other interpolation algorithms: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    # cv.INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results.
    # But when the image is zoomed, it is similar to the cv.INTER_NEAREST method.
    Im_resized = cv.resize(Im, (image_dim, image_dim), interpolation = cv.INTER_AREA) # INTER_LINEAR - bilinear interpolation
    # Convert to greyscale
    Im = cv.cvtColor(Im_resized, cv.COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(f"Resized (& grayscale) version of {os.path.basename(filepath)}")
    print("\nResized (& grayscale) image size: ", image_dim, image_dim)

    # Perform the DCT2 Transform
    DCT_128 = NPointDCT2(image_dim)
    result = np.matmul(np.matmul(DCT_128, Im.astype(float)), np.transpose(DCT_128))
    
    # Display the transformed image
    plt.figure()
    plt.imshow(result, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title(f"DCT2 Transform of {os.path.basename(filepath)}")
    print("\nFinal result size: ", result.shape)

    # Show figures
    plt.show()


def DaubechiesWaveletTransform(size):
    # This code calculates the Daubechies-4 wavelet coefficients.
    
    h = [(1 + np.sqrt(3)) / 4, (3 + np.sqrt(3)) / 4, (3 - np.sqrt(3)) / 4, (1 - np.sqrt(3)) / 4]
    g = [-h[3], h[2], -h[1], h[0]]
    
    Daub4 = np.zeros((size, size))
    j = 0
    
    for i in range(size // 2):
        Daub4[i, j] = h[0]
        Daub4[i, (j + 1) % size] = h[1]
        Daub4[i, (j + 2) % size] = h[2]
        Daub4[i, (j + 3) % size] = h[3]
        
        Daub4[i + size // 2, j] = g[0]
        Daub4[i + size // 2, (j + 1) % size] = g[1]
        Daub4[i + size // 2, (j + 2) % size] = g[2]
        Daub4[i + size // 2, (j + 3) % size] = g[3]
        
        j = (j + 2) % size
    
    return Daub4


def testDaubechiesWavelet(filepath):
    # Testing Daubechies Wavelet Transform (using squareX.bmp)
    
    # Load and display the original image (replace 'squareX.bmp' with your image)
    # Load the image
    Im = plt.imread(filepath)
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title("squareX.bmp")

    row, col = Im.shape
    IN = Im.astype(float)

    Daub4 = DaubechiesWaveletTransform(row)
    IDTemp = np.zeros((row, col))

    for i in range(row):
        IDTemp[i, :] = (1 / np.sqrt(2)) * np.dot(Daub4, IN[i, :])

    IDaub = np.zeros((row, col))

    for j in range(col):
        IDaub[:, j] = (1 / np.sqrt(2)) * np.dot(Daub4, IDTemp[:, j])
    
    # More
    # Extract the top-left quarter of the image and apply the Daubechies4 transform
    Daub4 = DaubechiesWaveletTransform(row // 2)
    IN = IDaub[:row // 2, :col // 2]
    IDTemp = np.zeros((row // 2, col // 2))

    for i in range(row // 2):
        IDTemp[i, :] = (1 / np.sqrt(2)) * np.dot(Daub4, IN[i, :])
    
    IDaub2 = np.zeros((row // 2, col // 2))

    for j in range(col // 2):
        IDaub2[:, j] = (1 / np.sqrt(2)) * np.dot(Daub4, IDTemp[:, j])
    
    # Replace the top-left quarter of the original IDaub with the transformed quarter
    IDaub[:row // 2, :col // 2] = IDaub2

    plt.figure()
    plt.imshow(np.abs(IDaub), cmap='gray', vmin=0, vmax=255)
    plt.title("Daubechies Wavelet")
    
    # Show figures
    plt.show()


def main(debug):
    print("\n")

    # plotDCT2(EXAMPLE_IMAGE_COLOR_OBLONG, image_dim = 32)
    plotTwoDHaar(EXAMPLE_IMAGE_GREYSCALE, image_dim = 32)
    
    # testDaubechiesWavelet(EXAMPLE_IMAGE_GREYSCALE)


if __name__ == "__main__":
    main(debug = False)