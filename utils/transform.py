import numpy as np
import matplotlib.pyplot as plt

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/traffic_signs"
EXAMPLE_IMAGE = f"{ROOT_DIR}/utils/squareX.bmp"


def TwoDHaarTransform(Im, L):
    # This code is for educational and research purposes of comparisons.
    # This program was written to implement the Haar Wavelet on an image for 
    # the intention of demonstrating the Haar wavelet.  
    #
    # Inputs:
    #       Im is the image to be transformed.
    #       L is how many levels of decomposition are to be executed.
    #
    # Output:
    #       waveletIm is the transformed image coefficients from the subbands 
    #       {LL, LH, HL, HH} for each level

    waveletIm = Im.copy()
    
    for l in range(L):
        n1, n2 = Im.shape
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
                
                LL[i // 2, j // 2] = (im00 + im01 + im10 + im11) / 4
                LH[i // 2, j // 2] = (im00 + im01 - im10 - im11) / 4
                HL[i // 2, j // 2] = (im00 - im01 + im10 - im11) / 4
                HH[i // 2, j // 2] = (im00 - im01 - im10 + im11) / 4
        
        synImage = np.zeros((n1, n2), dtype=float)
        synImage[0:n1 // 2, 0:n2 // 2] = LL
        synImage[n1 // 2:n1, 0:n2 // 2] = LH
        synImage[0:n1 // 2, n2 // 2:n2] = HL
        synImage[n1 // 2:n1, n2 // 2:n2] = HH
        
        Im[0:n1 // 2, 0:n2 // 2] = LL
        waveletIm = synImage if l == 0 else np.dstack((waveletIm, synImage))
    
    return waveletIm


def testTwoDHaar(filepath):
    # Testing 2D Haar Wavelet Transform (using squareX.bmp)

    # Load and display the original image (replace 'squareX.bmp' with your image)
    Im = plt.imread(filepath)
    plt.figure()
    plt.imshow(Im, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title("squareX.bmp")

    # Perform the 2D Haar Wavelet Transform
    L = 4
    waveletIm = TwoDHaarTransform(Im.astype(float), L)
    
    # Display the transformed image
    plt.figure()
    plt.imshow(np.abs(waveletIm), cmap = 'gray', vmin = 0, vmax = 255)
    plt.title("2D Haar Wavelet Transform")

    # Show figures
    plt.show()


def main(debug):
    print("\n")
    testTwoDHaar(EXAMPLE_IMAGE)


if __name__ == "__main__":
    main(debug = False)