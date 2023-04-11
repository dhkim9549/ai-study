import numpy as np
class Tensor (object):

    def __init__(self,data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):

        self.data = np.array(data)
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if(id is None):
            id = np.random.randint(0,100000)
        self.id = id

        if(creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if(cnt != 0):
                return False
        return True

    def backward(self,grad=None, grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad

            if(self.creators is not None and
               (self.all_children_grads_accounted_for() or
                grad_origin is None)):

                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())

    def show(self, name=None):
        print(f'{name}: id = {self.id}, creators = {self.creators}, children = {self.children}, data = {self.data}, grad = {self.grad}')

a = Tensor([1,2,3,4,5], autograd=True, id='a')
b = Tensor([2,2,2,2,2], autograd=True, id='b')
c = Tensor([5,4,3,2,1], autograd=True, id='c')
a.show('a')
b.show('b')
c.show('c')

d = a + b
d.show('d')
a.show('a')
b.show('b')
e = b + c
f = d + e
f.show('f')
b.show('b')

f.backward(Tensor(np.array([1,1,1,1,1])))
f.backward(Tensor(np.array([1,1,1,1,1])))
f.show('f')
d.show('d')
e.show('e')
a.show('a')
b.show('b')
c.show('c')
