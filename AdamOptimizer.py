import torch

class SimpleAdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.ms = [torch.zeros_like(param) for param in self.params]
        self.vs = [torch.zeros_like(param) for param in self.params]

    def step(self, grads):
        self.t += 1
        for i, param in enumerate(self.params):
            grad = grads[i]
            m, v = self.ms[i], self.vs[i]
    
            # Update biased first and second moment estimates
            m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
    
            # Compute bias-corrected first and second moment estimates
            m_hat = m.div(1 - self.beta1 ** self.t)
            v_hat = v.div(1 - self.beta2 ** self.t)
    
            # Update the parameter using a copy operation
            param_update = m_hat.div(v_hat.sqrt().add_(self.eps)).mul(-self.lr)
            with torch.no_grad():
                param += param_update
                param.grad = None
