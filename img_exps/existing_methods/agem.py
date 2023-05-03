from img_exps.vision_utils import apply_mask
import torch
import numpy as np


def store_grad(pp, grads, grad_dims, tid):
    """
    This stores parameter gradients of past tasks.
    :param pp: parameters
    :param grads: gradients
    :param grad_dims: list with number of parameters per layers
    :param tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient vector, whenever violations occur.
    :param pp: parameters
    :param newgrad: new gradients to be updated
    :param grad_dims: list with number of parameters per layers
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


class AGEM(torch.nn.Module):
    """
    Implementation of A-GEM proposed in
            Chaudhry, Arslan, et al. "Efficient lifelong learning with a-gem."
            arXiv preprint arXiv:1812.00420 (2018).
    """

    def __init__(
        self,
        args,
        net,
        optimizer,
        criterion,
        classes_per_task,
        num_tasks,
        input_size=(3, 224, 224),
    ):
        super(AGEM, self).__init__()

        self.net = net
        self.ce = criterion
        self.batch_size = args.batch_size
        self.input_size = input_size
        self.opt = optimizer

        self.n_mem_per_class = args.mem_size
        self.nc_per_task = classes_per_task

        self.memory_data = torch.FloatTensor(
            num_tasks, self.nc_per_task, self.n_mem_per_class, *self.input_size
        ).cuda()
        self.memory_labs = torch.LongTensor(
            num_tasks, self.nc_per_task, self.n_mem_per_class
        ).cuda()

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), 2)
        self.grads = self.grads.cuda()

        self.observed_tasks = []
        self.old_task = -1
        self.class_counts = np.zeros((num_tasks, self.nc_per_task)).astype(int)

    def sample(self, mem_x, mem_y):
        if mem_y.size(0) < self.batch_size:
            return mem_x, mem_y
        else:
            indices = torch.from_numpy(
                np.random.choice(mem_x.size(0), self.batch_size, replace=False)
            )
            return mem_x[indices], mem_y[indices]

    def observe_agem(self, net, x, t, y):
        """
        Computing new gradients for the given input and labels
        :param net: Current model
        :param x: Input samples
        :param t: Current task
        :param y: Labels
        :return: Model with the new corresponding gradients
        """

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
        y_prev = y
        y = t * self.nc_per_task + y

        for i in range(len(x)):
            y_ind = (y.data[i] % self.nc_per_task).cpu().numpy()
            self.memory_data[t, y_ind, self.class_counts[t, y_ind]].copy_(x.data[i])
            self.memory_labs[t, y_ind, self.class_counts[t, y_ind]].copy_(y.data[i])
            self.class_counts[t, y_ind] += 1
            if self.class_counts[t, y_ind] == self.n_mem_per_class:
                self.class_counts[t, y_ind] = 0

        if len(self.observed_tasks) > 1:
            self.zero_grad()
            prev_tasks = self.observed_tasks[:-1]
            mem_x, mem_y = self.sample(
                self.memory_data[prev_tasks].reshape(
                    (
                        len(prev_tasks * self.n_mem_per_class * self.nc_per_task),
                        *self.input_size,
                    )
                ),
                self.memory_labs[prev_tasks].reshape(-1),
            )
            mem_preds = net(mem_x, None)
            mem_preds = apply_mask(mem_y, mem_preds, self.nc_per_task)
            # print(self.memory_data[prev_tasks].shape, self.memory_labs[prev_tasks], mem_y, torch.argmax(mem_preds,dim=1))
            ptloss = self.ce(mem_preds, mem_y)
            ptloss.backward()
            store_grad(net.parameters, self.grads, self.grad_dims, 1)

        self.zero_grad()

        pred = net(x, t)
        loss = self.ce(pred, y_prev)
        loss.backward()

        if len(self.observed_tasks) > 1:
            store_grad(net.parameters, self.grads, self.grad_dims, 0)
            dotp = torch.dot(self.grads[:, 0], self.grads[:, 1])
            if dotp < 0:
                self.grads[:, 0] -= (
                    dotp / torch.dot(self.grads[:, 1], self.grads[:, 1])
                ) * self.grads[:, 1]
                overwrite_grad(net.parameters, self.grads[:, 0], self.grad_dims)
        return net