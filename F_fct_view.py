import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.fftpack import fftshift, fft2
import pyvista as pv
import seaborn as sns

# ----------------------------------------------------- #
def Gaussion_spectral_energy_map():
    N = 512 
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(x, y)
    sigma = 2.0 
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    fft_gaussian = np.fft.fftshift(np.fft.fft2(gaussian))

    spectrum_energy = np.abs(fft_gaussian)**2

    plt.figure(figsize=(8, 6))
    plt.imshow(np.log10(spectrum_energy + 1e-10), cmap='viridis', extent=[-N/2, N/2, -N/2, N/2])
    plt.colorbar(label='Log10(Energy)')
    plt.title('Frequency Spectrum Energy')
    plt.xlabel('Frequency (k_x)')
    plt.ylabel('Frequency (k_y)')
    plt.show()



# ----------------------------------------------------- #
def frq_view_spectrum(z):
    # fft
    z_fft = torch.fft.rfftn(z, dim=[-3, -2, -1], norm="ortho")  # 三维FFT，输出复数张量

    # magnitude
    magnitude = torch.abs(z_fft)

    # energy
    # power = magnitude ** 2  # all
    power = magnitude

    mean_power_c = power.mean(dim=1)  # channle mean

    mean_power_b = power.mean(dim=0)  # batch mean

    return mean_power_c,mean_power_b


def radial_profile_3d(spectrum):
    z, y, x = np.indices(spectrum.shape)
    center = np.array(spectrum.shape) // 2
    r = np.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2).astype(int)

    radial_sum = np.bincount(r.ravel(), spectrum.ravel())
    radial_count = np.bincount(r.ravel())
    return radial_sum / radial_count 

def radial_profile_3d_plot(original_spectrum,encoded_spectrum):
    r_original = radial_profile_3d(original_spectrum)
    r_encoded = radial_profile_3d(encoded_spectrum)

    # 可视化对比
    plt.plot(r_original, label='Original', color='blue')
    plt.plot(r_encoded, label='VAE Reconstructed', color='red', linestyle='--')
    plt.axvline(x=12, color='gray', linestyle=':')  
    plt.xlabel('Spatial Frequency Radius')
    plt.ylabel('Normalized Energy')
    plt.yscale('log') 
    plt.legend()
    plt.show()



def plot_ortho_slices(spectrum, vmax=None):
    # vmax = spectrum.max() * 0.1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slices = [
        spectrum[spectrum.shape[0] // 2, :, :], 
        spectrum[:, spectrum.shape[1] // 2, :], 
        spectrum[:, :, spectrum.shape[2] // 2]  
    ]

    titles = ['XY Slice', 'XZ Slice', 'YZ Slice']
    for ax, slc, title in zip(axes, slices, titles):
        im = ax.imshow(np.log(slc + 1e-9),
                       cmap='viridis',
                       vmax=np.log(vmax) if vmax else None)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()



def plot_isosurface(spectrum, threshold=0.1):
    grid = pv.UniformGrid()
    grid.dimensions = np.array(spectrum.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = spectrum.flatten(order="F")

    contours = grid.contour([spectrum.max() * threshold])

    p = pv.Plotter()
    p.add_mesh(contours, color='orange', opacity=0.5)
    p.show()


# plot_isosurface(original_spectrum - reconstructed_spectrum) 
# ----------------------------------------------------- #

def process_binary_image(image, operator_name):

    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image / 255  
    binary_image = binary_image.astype(np.uint8) 

    if operator_name == "morphology_edge":
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        edge = binary_image - eroded
        return edge * 255  

    elif operator_name == "sobel":
        sobel_x = cv2.Sobel(binary_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(binary_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        if np.max(sobel) == 0:
            sobel = np.zeros_like(sobel, dtype=np.uint8) 
        else:
            sobel = np.uint8(255 * sobel / np.max(sobel)) 
        return sobel

    elif operator_name == "canny":
        canny = cv2.Canny(binary_image, threshold1=100, threshold2=200)
        return canny

    elif operator_name == "lbp":
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(binary_image, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6) 
        return lbp, lbp_hist 

    elif operator_name == "glcm":
        distances = [1]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(binary_image * 255, distances=distances, angles=angles, levels=256, symmetric=True,
                            normed=True)
        contrast = graycoprops(glcm, 'contrast')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')
        correlation = graycoprops(glcm, 'correlation')
        features = {
            "contrast": contrast,
            "energy": energy,
            "homogeneity": homogeneity,
            "correlation": correlation
        }
        return glcm, features 

    elif operator_name == "fourier":

        f_transform = fft2(binary_image)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-6) 
        return magnitude_spectrum  

    elif operator_name == "gabor":

        kernels = [] 
        for theta in np.arange(0, np.pi, np.pi / 4):  
            for freq in (0.1, 0.2, 0.3): 
                kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=theta, lambd=1 / freq, gamma=0.5, psi=0,
                                            ktype=cv2.CV_32F)
                kernels.append(kernel)

        responses = []
        for kernel in kernels:
            filtered = cv2.filter2D(binary_image, cv2.CV_8UC3, kernel)
            responses.append(filtered)

        return responses 

    else:
        raise ValueError(f"Unsupported operator: {operator_name}")


if __name__ == "__main__":
    print()
