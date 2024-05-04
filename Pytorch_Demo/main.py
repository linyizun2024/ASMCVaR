import torch
import torch.autograd as autograd
import torch.optim as optim
import scipy.io
import os

global v, y, h1, h2, lbd, rho, N, N1, T, tilde_I, gamma, beta1, beta2, m, matQ, vecq, theta, kappa


def f_fuc(v, h1, h2, lbd, rho):
    return h1.T @ v + lbd * (h2.T @ v - rho) ** 2

def Sm_func(y, m):
    vals, indices = y.topk(m, dim=0, largest=True)
    y.data.zero_()  
    y.data.scatter_(0, indices, vals)
    return y

def FPPA(v, matQ, vecq, theta, kappa, max_iter=2, tol=0.001):
    x = matQ @ v
    for iter_ in range(max_iter):
        x_pre = x.clone()
        q_tmp = matQ @ v + x_pre - theta * (matQ @ (matQ.T @ x_pre))
        x = (1 - kappa) * x_pre + kappa * q_tmp - kappa * torch.max(q_tmp, vecq)
        RE = torch.norm(x - x_pre) / torch.norm(x_pre)
        if RE <= tol:
            break
    v.data -= theta * (matQ.T @ x)
    return v

class H_func(autograd.Function):
    
    @staticmethod
    def forward(ctx, v, y, h1, h2, lbd, rho, tilde_I, gamma):
        diff = tilde_I @ v - y
        out = f_fuc(v, h1, h2, lbd, rho) + diff.T @ diff / (2 * gamma)
        ctx.save_for_backward(v, y, h1, h2, lbd, rho, tilde_I, gamma, diff)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        v, y, h1, h2, lbd, rho, tilde_I, gamma, diff = ctx.saved_tensors
        grad_tmp_v = h1 + 2 * lbd * (h2.T @ v - rho) * h2 + tilde_I.T @ diff / gamma
        grad_tmp_y = - diff / gamma
        v_grad = grad_output * grad_tmp_v
        y_grad = grad_output * grad_tmp_y
        return v_grad, y_grad, None, None, None, None, None, None

# Define loss function
def loss_function(v, y, h1, h2, lbd, rho, tilde_I, gamma):
    return H_func.apply(v, y, h1, h2, lbd, rho, tilde_I, gamma)

# Create parameter list
parameters_v = [v]
parameters_y = [y]

# Create optimizer
optimizer_v = optim.Adam(parameters_v, lr=beta1)
optimizer_y = optim.Adam(parameters_y, lr=beta2)  

for i in range(10):
    
    optimizer_v.zero_grad()
    optimizer_y.zero_grad()

    loss = loss_function(v, y, h1, h2, lbd, rho, tilde_I, gamma)
    loss.backward(retain_graph=True)
    
    optimizer_v.step()
    v = FPPA(v, matQ, vecq, theta, kappa)
    
    optimizer_y.step()
    y = Sm_func(y, m)

print(v)
print(y)

