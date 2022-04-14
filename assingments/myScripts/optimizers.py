import torch

class SGD_w_L1(torch.optim.Optimizer):
    def __init__(self,parameters : list, learning_rate : float, l : float = 0.001):
        super().__init__( parameters, {
            "lambda" : l,
            "learning_rate" : learning_rate,
        })
        #hyper parameters are passed to parent class as a dict
    @torch.no_grad()
    def step(self):
        for pg in self.param_groups: # note: param_groups is already here bc it's defined in the father class
            for param in pg["params"]:
                if param.grad is not None: # param update operated only on those params having gradient
                    param.sub_(param.grad.mul_(pg['learning_rate']))
                    #this piece accounts for the l1 penalization on the loss
                    #actually in the gradient you have one piece coming from the loss
                    # + one piece coming from the derivative of the L1 norm of the paramenters
                    # | theta | = sum [ \abs |theta|_i]
                    # grad( |theta|_l1 ) = [ sign(theta_1), ... , sign(theta_n)]
                    # this has to be multplied by the learning rate and the penalization term 
                    # then you put all togheter with the effect of the gradient
                    param.sub_(pg['lambda']*pg['learning_rate']*torch.sign(param))