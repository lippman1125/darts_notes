import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    """
    我们更新梯度就是theta = theta + v + weight_decay * theta
      1.theta就是我们要更新的参数
      2.weight_decay*theta为正则化项用来防止过拟合
      3.v的值我们分带momentum和不带momentum：
        普通的梯度下降：v = -dtheta * lr 其中lr是学习率，dx是目标函数对x的一阶导数
        带momentum的梯度下降：v = lr*(-dtheta + v * momentum)
    """

  # [完全复制外面的Network更新w的过程], 对应公式6第一项的w − ξ*dwLtrain(w, α)
  # 不直接用外面的optimizer来进行w的更新, 而是自己新建一个unrolled_model展开, 主要是因为我们这里的更新不能对Network的w进行更新
  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    # 把参数整理成一行代表一个参数的形式,得到我们要更新的参数theta
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # torch.autograd.grad 自动求导数
    # 前面的是loss对参数theta求梯度，self.network_weight_decay*theta就是正则项对参数进行更新，等价于optimizer.step()
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    # w − ξ*dwLtrain(w, α)
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # 计算公式六：dαLval(w',α) ，其中w' = w − ξ*dwLtrain(w, α)
    # w'
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    # Lval
    # 对做了一次更新后的w的unrolled_model求验证集的损失，Lval，以用来对α进行更新
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    # dαLval(w',α)
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    # dw'Lval(w',α)
    vector = [v.grad.data for v in unrolled_model.parameters()]
    # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    # 其中w+=w+dw'Lval(w',α)*epsilon w-=w-dw'Lval(w',α)*epsilon
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    # 公式六减公式八 dαLval(w',α)-(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
    for g, ig in zip(dalpha, implicit_grads):
      # In-place version of sub()
      g.data.sub_(eta, ig.data)

    # 对α进行更新
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  # 计算公式八(dαLtrain(w+,α)-dαLtrain(w-,α))/(2*epsilon)
  # 其中w+=w+dw'Lval(w',α)*epsilon w- = w-dw'Lval(w',α)*epsilon
  # vector就是dw'Lval(w',α)
  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    # dαLtrain(w+,α)
    for p, v in zip(self.model.parameters(), vector):
      # 将模型中所有的w'更新成w+=w+dw'Lval(w',α)*epsilon
      # In-place version of add()
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # dαLtrain(w-,α)
    for p, v in zip(self.model.parameters(), vector):
      # 将模型中所有的w'更新成w- = w+ - (w-)*2*epsilon = w+dw'Lval(w',α)*epsilon - 2*epsilon*dw'Lval(w',α)=w-dw'Lval(w',α)*epsilon
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    # 将模型的参数从w-恢复成w
    for p, v in zip(self.model.parameters(), vector):
      # w=(w-) +dw'Lval(w',α)*epsilon = w-dw'Lval(w',α)*epsilon + dw'Lval(w',α)*epsilon = w
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

