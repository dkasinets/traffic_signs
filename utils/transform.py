import numpy as np
import matplotlib.pyplot as plt

# Global Variables
ROOT_DIR = "/Users/Kasinets/Dropbox/Mac/Desktop/SP22_JHU/Rodriguez/TRAFFIC_SIGNS/traffic_signs"
EXAMPLE_IMAGE = f"{ROOT_DIR}/utils/squareX.bmp"


def TwoDHaarTransform(Im, L):
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
                im01 = Im[i + 1, j]
                im10 = Im[i, j + 1]
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


# TODO: Work on DCT instead
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
    testTwoDHaar(EXAMPLE_IMAGE)
    # testDaubechiesWavelet(EXAMPLE_IMAGE)


if __name__ == "__main__":
    main(debug = False)