import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            distances = distances#*torch.tensor(100) # soft-hard weight
            #print(distances)
            soft_distances = F.softmax(distances*torch.tensor(-1),dim=1)
            # hard_distances = F.softmax(distances*torch.tensor(-10000),dim=1)
            # print(soft_distances)
            # print("---aaa------")
            # print(hard_distances)
            #print(soft_distances)
            # _, indices_flatten = torch.min(soft_distances, dim=1)
            # indices = indices_flatten.view(*inputs_size[:-1])
            #ctx.mark_non_differentiable(indices)

            return soft_distances

    @staticmethod
    def backward(ctx, grad_output1,grad_output2):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = inputs
        _, indices = torch.max(indices, dim=1)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(inputs,codebook)
        # ctx.mark_non_differentiable(indices_flatten)
        z_q_soft = (inputs @ codebook)
        z_q_hard = torch.index_select(codebook, dim=0,index=indices_flatten)
        # codes_flatten = torch.index_select(codebook, dim=0,
            # index=indices_flatten)
        return z_q_soft,z_q_hard

    @staticmethod
    def backward(ctx, grad_output,grad_output2):
        #print("cnm")
        grad_inputs, grad_codebook = None, None
        grad_output = grad_output2
        size,codebook = ctx.saved_tensors
        # print(indices.shape)
        # print(grad_output.shape)
        # print(grad_codebook)
        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = torch.mm(grad_output,codebook.t())
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            # grad_codebook = torch.zeros(torch.mm(indices.mean(dim=0).t(),grad_output).size(),dtype = torch.float32).cuda()
            grad_codebook =torch.mm(size.t(),grad_output)
        # print(indices.shape)
        # print(grad_output.shape)
        # print(grad_codebook)
        return grad_inputs,grad_codebook
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
class VectorQuantization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs, codebook):
        embedding_size = codebook.size(1)
        inputs_flatten = inputs.view(-1, embedding_size)
        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
        distances = torch.addmm(codebook_sqr + inputs_sqr,
            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
        distances = distances#*torch.tensor(100) # soft-hard weight
        soft_distances = F.softmax(distances*torch.tensor(-1),dim=1)
        return soft_distances
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.grads = {}
        self.nvq = VectorQuantization()
    def forward(self, z_e_x):
        latents = self.vq(z_e_x, self.embedding.weight)
        return latents
    def save_grad(self,name):
        def hook(grad):
            self.grads[name] = grad
        return hook
    def straight_through(self, z_e_x):
        indices_soft = self.nvq(z_e_x, self.embedding.weight.detach())
        z_q_x_soft,z_q_x_hard= vq_st(indices_soft, self.embedding.weight)
        # print(self.embedding.weight)
        z_q_x_soft = z_q_x_soft.view_as(z_e_x)
        z_q_x_hard = z_q_x_hard.view_as(z_e_x)
        #print("---------------------indec")
        #print(indices_hard)

        #indices_hard.register_hook(self.save_grad('hard_indices'))
        return z_q_x_soft,z_q_x_hard