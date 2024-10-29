import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

## from https://github.com/locuslab/orthogonal-convolutions
def cayley(W):
    if len(W.shape) == 2:
        return cayley(W[None])[0]
    _, cout, cin = W.shape
    if cin > cout:
        return cayley(W.transpose(1, 2)).transpose(1, 2)
    U, V = W[:, :cin], W[:, cin:]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)[None, :, :]
    A = U - U.conj().transpose(1, 2) + V.conj().transpose(1, 2) @ V
    iIpA = torch.inverse(I + A)
    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=1)

def fft_shift_matrix( n, s):
        shift = torch.arange(0, n).repeat((n, 1))
        shift = shift + shift.T
        return torch.exp(1j * 2 * np.pi * s * shift / n)

class StridedConv(nn.Module):
    def __init__(self, *args, **kwargs):
        striding = False
        if 'stride' in kwargs and kwargs['stride'] == 2:
            kwargs['stride'] = 1
            striding = True
        super().__init__(*args, **kwargs)
        downsample = "b c (w k1) (h k2) -> b (c k1 k2) w h"
        if striding:
            self.register_forward_pre_hook(lambda _, x: \
                    einops.rearrange(x[0], downsample, k1=2, k2=2))

class PaddingChannels(nn.Module):
    def __init__(self, ncin, ncout, scale=1.0):
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.scale = scale

    def forward(self, x):
        bs, _, size1, size2 = x.shape
        out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
        out[:, :self.ncin] = self.scale * x
        return out

class PaddingFeatures(nn.Module):
    def __init__(self, fin, n_features, scale=1.0):
        super().__init__()
        self.n_features = n_features
        self.fin = fin
        self.scale = scale

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.n_features, device=x.device)
        out[:, :self.fin] = self.scale * x
        return out

class PlainConv(nn.Conv2d):
    def forward(self, x):
        return super().forward(F.pad(x, (1,1,1,1)))

class Plain_normed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, input):

        with torch.no_grad():
            sigma =  torch.linalg.matrix_norm(self.lin.weight, 2)
        if sigma>1:
            snl = torch.nn.utils.parametrizations.spectral_norm(self.lin)
            out = snl(input)
        else:
            out = self.lin(input)

        return out
class LinearNormalized(nn.Linear):

  def __init__(self, in_features, out_features, bias=True, scale=1.0):
    super(LinearNormalized, self).__init__(in_features, out_features, bias)
    self.scale = scale

  def forward(self, x):
    self.Q = F.normalize(self.weight, p=2, dim=1)
    return F.linear(self.scale * x, self.Q, self.bias)

class FirstChannel(nn.Module):
    def __init__(self, cout, scale=1.0):
        super().__init__()
        self.cout = cout
        self.scale = scale

    def forward(self,x):
        xdim = len(x.shape)
        if xdim == 4:
            return self.scale * x[:,:self.cout,:,:]
        elif xdim == 2:
            return self.scale * x[:,:self.cout]

class SandwichLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, AB=False):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.AB = AB
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B @ x
        if self.AB:
            x = 2 * F.linear(x, Q[:, :fout].T) # 2 A.T @ B @ x
        if self.bias is not None:
            x += self.bias
        return x

class SandwichFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features+out_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.psi = nn.Parameter(torch.zeros(out_features, dtype=torch.float32, requires_grad=True))
        self.Q = None

    def forward(self, x):
        fout, _ = self.weight.shape
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        x = F.linear(self.scale * x, Q[:, fout:]) # B*h
        if self.psi is not None:
            x = x * torch.exp(-self.psi) * (2 ** 0.5) # sqrt(2) \Psi^{-1} B * h
        if self.bias is not None:
            x += self.bias
        x = F.relu(x) * torch.exp(self.psi) # \Psi z
        x = 2 ** 0.5 * F.linear(x, Q[:, :fout].T) # sqrt(2) A^top \Psi z
        return x

class SandwichConv1(nn.Module):
    def __init__(self,cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:,:, None, None])
        x += self.bias[:, None, None]
        return F.relu(x)

class SandwichConv1Lin(nn.Module):
    def __init__(self,cin, cout, scale=1.0) -> None:
        super().__init__()
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin+cout))
        self.bias = nn.Parameter(torch.empty(cout))
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.kernel.norm()
        self.Q = None

    def forward(self, x):
        cout = self.kernel.shape[0]
        if self.training or self.Q is None:
            P = cayley(self.alpha * self.kernel / self.kernel.norm())
            self.Q = 2 * P[:, :cout].T @ P[:, cout:]
        Q = self.Q if self.training else self.Q.detach()
        x = F.conv2d(self.scale * x, Q[:,:, None, None])
        x += self.bias[:, None, None]
        return x

class SandwichConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args[0] += args[1]
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        xfft = 2 * Qfft[:,:,:cout].conj().transpose(1,2) @ Qfft[:,:,cout:] @ xfft
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]

        return x

class SandwichConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2) # //2 kernel_size; optional
                kwargs['padding'] = args[2] // 2 # TODO: added maxes recently
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args[0] += args[1]
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.psi  = nn.Parameter(torch.zeros(args[1]))
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, chn, _, _ = self.weight.shape
        cin = chn - cout
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)

        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, chn, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())

        Qfft = self.Qfft if self.training else self.Qfft.detach()
        # Afft, Bfft = Qfft[:,:,:cout], Qfft[:,:,cout:]
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        xfft = 2 ** 0.5 * torch.exp(-self.psi).diag().type(xfft.dtype) @ Qfft[:,:,cout:] @ xfft
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))
        if self.bias is not None:
            x += self.bias[:, None, None]
        xfft = torch.fft.rfft2(F.relu(x)).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cout, batches)
        xfft = 2 ** 0.5 * Qfft[:,:,:cout].conj().transpose(1,2) @ torch.exp(self.psi).diag().type(xfft.dtype) @ xfft
        x = torch.fft.irfft2(xfft.reshape(n, n // 2 + 1, cout, batches).permute(3, 2, 0, 1))

        return x

## Orthogonal layer, from https://github.com/locuslab/orthogonal-convolutions

class OrthogonLin(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None

    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        y = F.linear(self.scale * x, Q, self.bias)
        return y

class OrthogonFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0):
        super().__init__(in_features, out_features, bias)
        self.activation = nn.ReLU(inplace=False)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True))
        self.alpha.data = self.weight.norm()
        self.scale = scale
        self.Q = None

    def forward(self, x):
        if self.training or self.Q is None:
            self.Q = cayley(self.alpha * self.weight / self.weight.norm())
        Q = self.Q if self.training else self.Q.detach()
        y = F.linear(self.scale * x, Q, self.bias)
        y = self.activation(y)
        return y

class OrthogonConvLin(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2)
                kwargs['padding'] = args[2] // 2
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        return y

class OrthogonConv(StridedConv, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        args = list(args)
        if 'stride' in kwargs and kwargs['stride'] == 2:
            args = list(args)
            args[0] = 4 * args[0] # 4x in_channels
            if len(args) == 3:
                args[2] = max(1, args[2] // 2)
                kwargs['padding'] = args[2] // 2
            elif 'kernel_size' in kwargs:
                kwargs['kernel_size'] = max(1, kwargs['kernel_size'] // 2)
                kwargs['padding'] = kwargs['kernel_size'] // 2
        scale = 1.0
        if 'scale' in kwargs:
            scale = kwargs['scale']
            del kwargs['scale']
        args = tuple(args)
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.activation = nn.ReLU(inplace=False)
        self.register_parameter('alpha', None)
        self.Qfft = None

    def forward(self, x):
        x = self.scale * x
        cout, cin, _, _ = self.weight.shape
        batches, _, n, _ = x.shape
        if not hasattr(self, 'shift_matrix'):
            s = (self.weight.shape[2] - 1) // 2
            self.shift_matrix = fft_shift_matrix(n, -s)[:, :(n//2 + 1)].reshape(n * (n // 2 + 1), 1, 1).to(x.device)
        xfft = torch.fft.rfft2(x).permute(2, 3, 1, 0).reshape(n * (n // 2 + 1), cin, batches)
        if self.training or self.Qfft is None or self.alpha is None:
            wfft = self.shift_matrix * torch.fft.rfft2(self.weight, (n, n)).reshape(cout, cin, n * (n // 2 + 1)).permute(2, 0, 1).conj()
            if self.alpha is None:
                self.alpha = nn.Parameter(torch.tensor(wfft.norm().item(), requires_grad=True).to(x.device))
            self.Qfft = cayley(self.alpha * wfft / wfft.norm())
        Qfft = self.Qfft if self.training else self.Qfft.detach()
        yfft = (Qfft @ xfft).reshape(n, n // 2 + 1, cout, batches)
        y = torch.fft.irfft2(yfft.permute(3, 2, 0, 1))
        if self.bias is not None:
            y += self.bias[:, None, None]
        y = self.activation(y)
        return y

# SDP Lipschitz Layer, from https://github.com/araujoalexandre/lipschitz-sll-networks
class SLLBlockConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, scale=1.0, epsilon=1e-6):
        super().__init__()

        self.activation = nn.ReLU(inplace=False)
        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.randn(cout))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, bias=self.bias, padding=1)
        res = self.activation(res)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        q_abs = torch.abs(self.q)
        T = 2 / (torch.abs(q_abs[None, :, None, None] * kkt).sum((1, 2, 3)) / q_abs)
        res = T[None, :, None, None] * res
        res = F.conv_transpose2d(res, self.kernel, padding=1)
        out = x - res
        return out

class SLLBlockFc(nn.Module):
    def __init__(self, cin, cout, scale = 1.0, epsilon=1e-6):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(self.scale * x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1/(q_abs+self.epsilon))[:, None]
        T = 2/torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out

# almost orthogonal layer (AOL), based on SLL implementation

class AolLin(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x):
        T = 1/torch.sqrt(torch.abs(self.weights.T @ self.weights).sum(1))
        x = self.scale * T * x
        res = F.linear(x, self.weights, self.bias)
        return res

class AolFc(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x):
        T = 1/torch.sqrt(torch.abs(self.weights.T @ self.weights).sum(1))
        x = self.scale * T * x
        res = F.linear(x, self.weights, self.bias)
        return F.relu(res)

class AolConvLin(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, epsilon=1e-6, scale=1.0):
        super().__init__()

        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.padding = (kernel_size - 1) // 2
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, padding=self.padding)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        T = 1 / torch.sqrt(torch.abs(kkt).sum((1, 2, 3)))
        res = T[None, :, None, None] * res + self.bias[:, None, None]
        return res

class AolConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, epsilon=1e-6, scale=1.0):
        super().__init__()

        self.scale = scale
        self.kernel = nn.Parameter(torch.empty(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(cout))
        self.padding = (kernel_size - 1) // 2
        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(self.scale * x, self.kernel, padding=self.padding)
        kkt = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        T = 1 / torch.sqrt(torch.abs(kkt).sum((1, 2, 3)))
        res = T[None, :, None, None] * res + self.bias[:, None, None]
        return F.relu(res)

#from https://github.com/ruigangwang7/StableNODE

def cayley2(W: torch.Tensor) -> torch.Tensor:
    cout, cin = W.shape
    if cin > cout:
        return cayley2(W.T).T
    U, V = W[:cin, :], W[cin:, :]
    I = torch.eye(cin, dtype=W.dtype, device=W.device)
    A = U - U.T + V.T @ V
    iIpA = torch.inverse(I + A)

    return torch.cat((iIpA @ (I - A), -2 * V @ iIpA), axis=0)

class CayleyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.empty(1).fill_(
            self.weight.norm().item()), requires_grad=True)

        self.Q_cached = None

    def reset_parameters(self):
        std = 1 / self.weight.shape[1] ** 0.5
        nn.init.uniform_(self.weight, -std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

        self.Q_cached = None

    def forward(self, X):
        if self.training:
            self.Q_cached = None
            Q = cayley2(self.alpha * self.weight / self.weight.norm())
        else:
            if self.Q_cached is None:
                with torch.no_grad():
                    self.Q_cached = cayley2(
                        self.alpha * self.weight / self.weight.norm())
            Q = self.Q_cached

        return F.linear(X, Q, self.bias)

class MonLipLayer(nn.Module):
    def __init__(self,
                 features: int,
                 unit_features: Sequence[int],
                 mu: float = 0.1,
                 nu: float = 10.):
        super().__init__()
        self.mu = mu
        self.nu = nu
        self.units = unit_features
        self.Fq = nn.Parameter(torch.empty(sum(self.units), features))
        nn.init.xavier_normal_(self.Fq)
        self.fq = nn.Parameter(torch.empty((1,)))
        nn.init.constant_(self.fq, self.Fq.norm())
        self.by = nn.Parameter(torch.zeros(features))
        Fr, fr, b = [], [], []
        nz_1 = 0
        for nz in self.units:
            R = nn.Parameter(torch.empty((nz, nz+nz_1)))
            nn.init.xavier_normal_(R)
            r = nn.Parameter(torch.empty((1,)))
            nn.init.constant_(r, R.norm())
            Fr.append(R)
            fr.append(r)
            b.append(nn.Parameter(torch.zeros(nz)))
            nz_1 = nz
        self.Fr = nn.ParameterList(Fr)
        self.fr = nn.ParameterList(fr)
        self.b = nn.ParameterList(b)
        # cached weights
        self.Q = None
        self.R = None
        self.act = ReHU(1)
    def forward(self, x):
        sqrt_gam = math.sqrt(self.nu - self.mu)
        sqrt_2 = math.sqrt(2.)
        if self.training:
            self.Q, self.R = None, None
            Q = cayley2(self.fq * self.Fq / self.Fq.norm())
            R = [cayley2(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
        else:
            if self.Q is None:
                with torch.no_grad():
                    self.Q = cayley2(self.fq * self.Fq / self.Fq.norm())
                    self.R = [cayley2(fr * Fr / Fr.norm()) for Fr, fr in zip(self.Fr, self.fr)]
            Q, R = self.Q, self.R

        xh = sqrt_gam * x @ Q.T
        yh = []
        hk_1 = xh[..., :0]
        idx = 0
        for k, nz in enumerate(self.units):
            xk = xh[..., idx:idx+nz]
            #gh = sqrt_2 * self.act (sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.b[k]) @ R[k]
            gh = sqrt_2 * F.relu (sqrt_2 * torch.cat((xk, hk_1), dim=-1) @ R[k].T + self.b[k]) @ R[k]
            hk = gh[..., :nz] - xk
            gk = gh[..., nz:]
            yh.append(hk_1-gk)
            idx += nz
            hk_1 = hk
        yh.append(hk_1)

        yh = torch.cat(yh, dim=-1)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh @ Q) + self.by

        return y

class BiLipNet(nn.Module):
    def __init__(self,
                 features: int,
                 unit_features: Sequence[int],
                 mu: float = 0.1,
                 nu: float = 10.,
                 nlayer: int = 1):
        super().__init__()
        self.nlayer = nlayer
        mu = mu ** (1./nlayer)
        nu = nu ** (1./nlayer)
        olayer = [CayleyLinear(features, features) for _ in range(nlayer+1)]
        self.orth_layers = nn.Sequential(*olayer)
        mlayer = [MonLipLayer(features, unit_features, mu, nu) for _ in range(nlayer)]
        self.mon_layers = nn.Sequential(*mlayer)

    def forward(self, x):
        for k in range(self.nlayer):
            x = self.orth_layers[k](x)
            x = self.mon_layers[k](x)
        x = self.orth_layers[self.nlayer](x)
        return x
