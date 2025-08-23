import torch

rdo = 2e3
clip = 1

# Modify to use different paths.
gamma_table = torch.load("backslash_data/gamma_table.pt")
r_gamma_table = torch.load("backslash_data/r_gamma_table.pt")

# Modify to use GPU if available.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def backslash(model):
    with torch.no_grad():
        # Evaluate the shape parameter
        n, var, mean = 0, 0, 0
        for param in model.parameters():
            param = param.flatten().detach()
            n += param.shape[0]
            var += torch.sum((param ** 2).to(device))
            mean += torch.sum(torch.abs(param).to(device))
        r_gamma = (n * var / mean ** 2).to(device=torch.device("cpu"))
        pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))
        shape = gamma_table[pos]
        std = torch.sqrt(var / n)
        n = torch.tensor(n)

        # Rate Constrained Optimization
        for param in model.parameters():
            constant = rdo * shape / n * torch.sign(param.data)
            param_reg = torch.pow(
                torch.abs(param.data) + clip, shape - 1)
            param.data -= constant * param_reg
        distribution = {"shape": shape, "standard": std}
    return distribution
