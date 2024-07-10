# gauss_surf.py
# Port of gauss_surf.m
# Adapted from A First Course in Machine Learning, Chapter 2.
#              Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# Surface and contour plots of a bivariate multivariate Gaussian
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D


def plot_axis(ax, xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5):
    """
    Helper fn to draw axes.
    :param ax: plot axis object
    :param xmin: x minimum
    :param xmax: x maximum
    :param ymin: y minimum
    :param ymax: y maximum
    :return:
    """
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    ax.grid(True, which='both')

    # set the x-spine
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    ax.xaxis.set_label_coords(1, 0.5)
    ax.yaxis.set_label_coords(0.5, 0)


def gaussian(mu, sigma, xmin, xmax, ymin, ymax, resolution):
    """
    Computes the mesh of values for a bivariate (2-dimensional) Gaussian pdf:
      $p(\mathbf{x}|\mu,\Sigma) =
          \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right\}$

    :param mu: bivariate Gaussian mean (2-element array)
    :param sigma: bivariate Gaussian covariance (2x2 array)
    :param xmin: mesh x-minimum
    :param xmax: mesh x-maximum
    :param ymin: mesh y-minimum
    :param ymax: mesh y-maximum
    :param resolution: resolution of the mesh
    :return: pdf values at each mesh point, X-mesh coodinates, Y-mesh coordinates
    """

    # Define the grid for visualisation
    #   X repeats rows that are in range -5 to 5
    #   Y repeats cols that are in range -5 to 5
    X, Y = numpy.meshgrid(numpy.linspace(xmin, xmax, resolution),
                          numpy.linspace(ymin, ymax, resolution))

    # ----- Compute the bivariate Gaussian -----
    c1 = 1.0/(2.0 * numpy.pi)
    c2 = 1.0/numpy.sqrt(numpy.linalg.det(sigma))

    # The following does these things:
    # (1) The X.flatten (same for Y) makes it so that the 2-dimensional
    #     array X values are serialized into one 1-dimensional array
    #     of length shape[0]*shape[1].
    # (2) The subtraction of the scalar mu[{0,1}] centers each vector
    # (3) stack along axis=1 creates a 2500 x 2 matrix, where each row
    #     is a corresponding x,y pair
    # altogether, this make it possible to compute the difference of
    #   each x,y pair from the mean, and then in the computation of the
    #   'exponent' below, we can simultaneously compute the dot products
    #   of all of the centered x,y pairs with the covariance matrix.
    diff = numpy.stack((X.flatten() - mu[0],
                        Y.flatten() - mu[1]),
                       axis=1)

    exponent = -0.5 * numpy.diag(numpy.dot(numpy.dot(diff,
                                                     numpy.linalg.inv(sigma)),
                                           diff.T))
    pdfv = c1 * c2 * numpy.exp(exponent)

    # now reshape the resulting pdfv (currently a single dimensional
    #   vector) according to the original shape of X (and Y)
    pdfv = numpy.reshape(pdfv, X.shape)

    return pdfv, X, Y


def plot_gaussian(pdfv, X, Y):
    # Contour plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_axis(ax,
              xmin=numpy.min(X), xmax=numpy.max(X),
              ymin=numpy.min(Y), ymax=numpy.max(Y))

    plt.contour(X, Y, pdfv)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    # Surface plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, pdfv, rstride=1, cstride=1,
                           cmap=matplotlib.cm.jet,
                           linewidth=0, antialiased=False)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    fig.colorbar(surf)

    plt.pause(.1)  # required on some systems so that rendering can happen

    plt.show()


def main():
    # Define the Gaussian parameters
    mu = numpy.array([1, 2])
    sigma = numpy.array([[2, 0.8], [0.8, 4]])

    # resolution for plot (larger -> slower rendering)
    resolution = 50

    pdf_values, X, Y = gaussian(mu, sigma,
                                xmin=-5, xmax=5,
                                ymin=-5, ymax=5,
                                resolution=resolution)

    plot_gaussian(pdf_values, X, Y)


if __name__ == "__main__":
    main()
