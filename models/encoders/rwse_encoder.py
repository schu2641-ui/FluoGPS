import torch
import torch.nn as nn

KERNEL = 'RWSE'  # [1, 2, ..., 16]
DIM_PE = 20
N_LAYERS = 2
MODEL_TYPE = 'linear'
NORM_TYPE = 'batchnorm'  # 'mlp' or 'linear'
class RWSEEncoder(torch.nn.Module):

      # Instantiated type of the KernelPE, e.g. RWSE
    def __init__(self, dim_in, dim_emb, pass_as_var=True, expand_x=True,
                 rwse_steps=None):
        super().__init__()
        self.kernel_type = KERNEL
        num_rw_steps = len(rwse_steps)  # Number of random walk steps used
        self.dim_pe = DIM_PE
        self.n_layers = N_LAYERS
        model_type = MODEL_TYPE.lower()  # Encoder NN model type for PEs
        norm_type = NORM_TYPE.lower()  # Raw PE normalization layer type
        self.pass_as_var = pass_as_var  # Whether to pass PE also as separate var
        if dim_emb - self.dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"PE dim size {self.dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - self.dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - self.dim_pe)
        self.expand_x = expand_x and dim_emb - self.dim_pe > 0

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if self.n_layers == 1:
                layers.append(nn.Linear(num_rw_steps, self.dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(num_rw_steps, 2 * self.dim_pe))
                layers.append(activation())
                for _ in range(self.n_layers - 2):
                    layers.append(nn.Linear(2 * self.dim_pe, 2 * self.dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * self.dim_pe, self.dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(num_rw_steps, self.dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(f"Precomputed '{pestat_var}' variable is "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_{self.kernel_type}.enable' to "
                             f"True, and also set 'posenc.kernel.times' values")

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch

