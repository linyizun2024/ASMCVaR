import torch
import torch.autograd as autograd
import torch.optim as optim
import scipy.io
import os

global matQ, vecq,h1, h2, lbd, rho, gamma, kappa, beta1, beta2, theta, N, T, tilde_I, m

# retrieve data
data_path = os.path.join(os.path.dirname(__file__),'Data')

#Load MATLAB file
v_data = scipy.io.loadmat(os.path.join(data_path, 'v.mat'))
y_data = scipy.io.loadmat(os.path.join(data_path, 'y.mat'))
mat_data = scipy.io.loadmat(os.path.join(data_path, 'matR.mat'))
matQ_data = scipy.io.loadmat(os.path.join(data_path, 'matQ.mat'))
vecq_data = scipy.io.loadmat(os.path.join(data_path, 'vecq.mat'))
h1_data = scipy.io.loadmat(os.path.join(data_path, 'vech1.mat'))
h2_data = scipy.io.loadmat(os.path.join(data_path, 'vech2.mat'))
param_data = scipy.io.loadmat(os.path.join(data_path, 'Param.mat'))
param_values = param_data['Param']

# Extract data
matR = torch.tensor(mat_data['matR'], dtype=torch.float32)
T, N = matR.size()
matQ = torch.tensor(matQ_data['matQ'], dtype=torch.float32)
vecq = torch.tensor(vecq_data['vecq'], dtype=torch.float32)
h1 = torch.tensor(h1_data['vech1'], dtype=torch.float32)
h2 = torch.tensor(h2_data['vech2'], dtype=torch.float32)
lbd = torch.tensor(float(param_values['lambda']), dtype=torch.float32)
rho = torch.tensor(float(param_values['rho']), dtype=torch.float32)
gamma = torch.tensor(float(param_values['gammacoe']), dtype=torch.float32)
kappa = torch.tensor(float(param_values['kappa']), dtype=torch.float32)
beta1 = torch.tensor(float(param_values['beta_1']), dtype=torch.float32)
beta2 = torch.tensor(float(param_values['beta_2']), dtype=torch.float32)
theta = torch.tensor(float(param_values['theta']), dtype=torch.float32)

m = 10
eye_N = torch.eye(N)
zeros_NTpuls1 = torch.zeros(N, T + 1)
tilde_I = torch.cat([eye_N, zeros_NTpuls1], 1)


def f_func(v, h1, h2, lbd, rho):
    return h1.T @ v + lbd * (h2.T @ v - rho) ** 2

def Sm_func(y, m):
    vals, indices = y.topk(m, dim=0, largest=True)
    y.data.zero_() 
    y.data.scatter_(0, indices, vals)
    return y

def FPPA(v, matQ, vecq, theta, kappa, max_iter=200, tol=0.001):
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
        out = f_func(v, h1, h2, lbd, rho) + diff.T @ diff / (2 * gamma)
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

v = torch.tensor(v_data['v'], dtype=torch.float32, requires_grad=True)
y = torch.tensor(y_data['y'], dtype=torch.float32, requires_grad=True)

# Create parameter list
parameters_v = [v]
parameters_y = [y]

# Create optimizer
optimizer_v = optim.Adam(parameters_v, lr=beta1)
optimizer_y = optim.Adam(parameters_y, lr=beta2)  

for i in range(10000):
    
    optimizer_v.zero_grad()
    optimizer_y.zero_grad()
    
    loss = loss_function(v, y, h1, h2, lbd, rho, tilde_I, gamma)
    loss.backward(retain_graph=True)
    
    #optimizer_v.step()
    v.data = v.data - beta1 * v.grad
    v = FPPA(v, matQ, vecq, theta, kappa)
    
    #optimizer_y.step()
    y.data = y.data - beta2 * y.grad
    y = Sm_func(y, m)
    
    # print("Gradient of x:", v.grad)
    # print("Gradient of y:", y.grad)

print(v)
print(y)

