import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import normalized_root_mse as NRMSE


def NCC(real_image, generated_image):
    """Method to compute the normalised cross correlation between two images.
    Arguments:
                real_image:       (numpy array) the real image
                predicted_image:  (numpy array) the predicted image by the model
    Returns:
                NCCScore:         (float) the normalised cross correlation score
    """
    # if the images are not the same size, raise an error
    if real_image.shape != generated_image.shape:
        raise AssertionError("The inputs must be the same size.")
    # reshape images to vectors
    u = real_image.reshape((real_image.shape[0] * real_image.shape[1] * real_image.shape[2], 1))
    v = generated_image.reshape((generated_image.shape[0] * generated_image.shape[1] * real_image.shape[2], 1))
    # take the real image and subtract the mean of the real image
    u = u - u.mean(keepdims=True)
    # take the generated image and subtract the mean of the generated image
    v = v - v.mean(keepdims=True)
    # transpose the real image for multiplication
    TransposedU = np.transpose(u)
    # calculate the length of the image
    length = np.linalg.norm(u, ord=2) * np.linalg.norm(v, ord=2)
    # calculate the NCC of the real image and the generated image
    NCCScore = float(TransposedU.dot(v)) / length
    # return the NCC score
    return NCCScore


def get_scores(real, gen):
    real = real - np.mean(real)
    gen = gen - np.mean(gen)
    ncc = NCC(real.squeeze(), gen.squeeze())
    ssim = SSIM(real.squeeze(), gen.squeeze(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    nrmse = NRMSE(real, gen)
    return ncc, ssim, nrmse
