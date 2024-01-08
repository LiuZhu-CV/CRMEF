import torch
class OutGradient(object):
    def __init__(self):
        pass
    def val_grad(self, val_loss_node, net_inner, iterations, net_outer,lr,gradients):
        def update_outer_gradient(gradients, alphas, lr, outer_gradients, net_outer):
            for g, a in zip(gradients, alphas):
                g = g * a
            # find the gradients
            ones_list = []
            for g in gradients:
                ones_list.append(torch.ones_like(g))
            p_del = torch.autograd.grad(gradients, net_outer.parameters(), grad_outputs=ones_list,retain_graph=True)
            for p, pd in zip(outer_gradients, p_del):
                p = p - lr * pd
            return outer_gradients

        def update_alpha(gradients, alphas, lr, outer_gradients, net_inner):

            # find the gradients
            ones_list = []
            for g in gradients:
                ones_list.append(torch.ones_like(g))
            p_del = torch.autograd.grad(gradients, net_inner.parameters(), grad_outputs=ones_list)
            for alpha, pd in zip(alphas, p_del):
                Iden = torch.ones_like(pd)
                alpha = alpha * (Iden - lr * pd)
            return alphas

        # val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
        #
        # val_output_ys = net_outer(val_inputs)
        # val_outputs = net_inner(val_output_ys)
        # val_loss_node = self.criterion(val_outputs, val_labels)
        # alpha and outer_gradient
        outer_gradient = torch.autograd.grad(val_loss_node,net_outer.parameters(),retain_graph=True,create_graph=True)
        alpha = torch.autograd.grad(val_loss_node,net_inner.parameters(),retain_graph=True,create_graph=True)
        for iter in range(iterations):
            index =iterations-iter-1
            gradient =gradients[index]
            outer_gradient = update_outer_gradient(gradient,alpha,lr,outer_gradient,net_outer)
            alpha = update_alpha(gradient,alpha,lr,outer_gradient,net_inner)
        return outer_gradient

