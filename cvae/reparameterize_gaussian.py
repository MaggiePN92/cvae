import torch


def reparameterize_gaussian(mean, std):
    """
    Inputs:
        mean : [torch.tensor] Mean vector. Shape: batch_size x z_dim.
        std  : [torch.tensor] Standard deviation vection. Shape: batch_size x z_dim.
    
    Output:
        z    : [torch.tensor] z sampled from the Normal distribution with mean and standard deviation given by the inputs. 
                              Shape: batch_size x z_dim.
    """
    # batch_size, z_dim = mean.shape
    # Sample epsilon from N(0,I)
    eps = torch.randn_like(mean)
    # eps = np.random.standard_normal((batch_size, z_dim))
    # eps = torch.from_numpy(eps)
    # Calculate z using reparameterization trick
    z = mean + (std * eps)
    return z 