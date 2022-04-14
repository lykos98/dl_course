from xmlrpc.client import Boolean
from matplotlib.pyplot import axis
from pyparsing import TokenConverter
from sklearn import metrics
import torch
import abc
import numpy as np
import os

def accuracy_at_k(nn_output: torch.Tensor, ground_truth: torch.Tensor, k: int=1):
    '''
    Return accuracy@k for the given model output and ground truth
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in accuracy@k
    '''
    assert k <= nn_output.shape[1], f"k too big. Found: {k}. Max: {nn_output.shape[1]} inferred from the nn_output"
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc

def accuracy(y,y_hat):
    y_c = y #torch.argmax(y, axis = 1)
    y_hat_c = torch.argmax(y_hat, axis = 1)

    return torch.sum(y_c == y_hat_c)/y_c.shape[0]


class Profiler(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models

    Improvement wrt the original class discussed in class, actually this keeps track and saves the full metric
    profile over the training
    '''
    def __init__(self, epochs : int) -> None:
        self.reset_state()
        self.avgs = np.array([0.  for _ in range(epochs)])
        self.current_epoch = 0
        self.extreme = 0

    def reset_state(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.batch_avg = 0

    @abc.abstractclassmethod
    def compare_for_extreme(self,a,b) -> Boolean:
        pass


    @abc.abstractmethod
    def update_epoch(self):
        pass

    def advance_epoch(self):
        self.avgs[self.current_epoch] = self.batch_avg

        if self.compare_for_extreme(self.batch_avg, self.extreme):
            #print("yep", self.extreme, self.batch_avg)
            self.extreme = self.batch_avg
            #print("yep", self.extreme, self.batch_avg)

        self.reset_state()
        self.current_epoch += 1

    def __repr__(self) -> str:
        return f"val {self.val} sum {self.sum} count {self.count} current_epoch {self.current_epoch}"
    
class LossProfiler(Profiler):
    def __init__(self, epochs: int) -> None:
        super().__init__(epochs)
        self.extreme = 100000000

    def update_epoch(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.batch_avg = self.sum / self.count
    
    def compare_for_extreme(self,a,b):
        return a < b

class MetricsProfiler(Profiler):
    def __init__(self, epochs: int, metrics : callable) -> None:
        """Contructor

        Args:
            epochs (int): number of epochs for which the model will be trained
            metric (callable): metric function to evaluate for each batch
        """
        super().__init__(epochs)
        self.metric_fn = metrics    
        self.extreme = 0

    def update_epoch(self, y, y_hat, n = 1):
        """Method to update measurements through batches during training

        Args:
            y (torch.tensor): Ground truth 
            y_hat (torch.tensor): Model predictions
        """
        self.val = self.metric_fn(y,y_hat)
        self.sum += self.val * n 
        self.count += n
        self.batch_avg = self.sum / self.count

    def compare_for_extreme(self,a,b):
        return a > b

def train_pass( model : torch.nn.Module ,
                dataloader :  torch.utils.data.DataLoader, 
                loss_fn : callable, 
                optimizer : torch.optim, 
                device : str, 
                loss_tracker : LossProfiler, 
                metric_tracker : MetricsProfiler,
                regularization_in_loss : bool = False):

    """Training pass function for one epoch of training

    Args:
        model (torch.nn.Module): _description_
        dataloader (torch.dataloader): _description_
        loss_fn (callable): _description_
        optimizer (torch.optim): _description_
        device (str): _description_
        loss_tracker (LossProfiler): _description_
        metric_tracker (MetricsProfiler): _description_
        regularization (_type_, optional): _description_. Defaults to None:str.
    """
  
    for x, y in dataloader:
        optimizer.zero_grad()
        x.to(device)
        y.to(device)
        y_hat = model(x)
       # y_hat.float()
        if not regularization_in_loss:
            loss = loss_fn(y_hat,y)
        else:
            loss = loss_fn(y_hat,y,model)
        loss.backward()
        optimizer.step()
        loss_tracker.update_epoch(loss.item(), n = y_hat.shape[0])
        metric_tracker.update_epoch(y,y_hat, n = y_hat.shape[0] )

def validation_pass(model, dataloader, loss_fn, device, loss_tracker, metric_tracker, regularization_in_loss = False):
    for x, y in dataloader:
        x.to(device)
        y.to(device)
        y_hat = model(x)
        if not regularization_in_loss:
            loss = loss_fn(y_hat,y)
        else:
            loss = loss_fn(y_hat,y,model)
        loss_tracker.update_epoch(loss.item(), n = y_hat.shape[0])
        metric_tracker.update_epoch(y,y_hat, n = y_hat.shape[0] )

def early_stopping_handler(criterion, model, path,tlt, tpt, vlt, vpt):
    if criterion == "Eopt":
#        print(vpt.extreme,vpt.avgs[vpt.current_epoch])
        if vpt.extreme == vpt.avgs[vpt.current_epoch - 1]:
            checkpoint_dict = {
                "parameters": model.state_dict(),
                "epoch": vpt.current_epoch
            }
            torch.save(checkpoint_dict, path) 
            print(f"Saved model at epoch: {vpt.current_epoch}")


def train(  model : torch.nn.Module, 
            train_dataloader : torch.utils.data.DataLoader, 
            validation_dataloader : torch.utils.data.DataLoader, 
            loss_fn : callable, 
            optimizer : torch.optim.Optimizer, 
            epochs : int, 
            validation_split : float = 0.8,
            device : str = "cuda", 
            performance : callable = accuracy, 
            checkpoint_location : str = None, 
            checkpoint_name : str = "cp.pt", 
            early_stopping : Boolean = False,
            early_stopping_criterion : str = "No",
            regularization_in_loss = None):

  
    if not checkpoint_location:
        os.makedirs("checkpoints", exist_ok=True) 
        checkpoint_location = "checkpoints"
    else:
        os.makedirs(checkpoint_location, exist_ok=True)

    train_loss_tracker = LossProfiler(epochs)
    train_performance_tracker = MetricsProfiler(epochs, metrics = performance)
    validation_loss_tracker = LossProfiler(epochs)
    validation_performance_tracker = MetricsProfiler(epochs, metrics = performance)

    for epoch in range(epochs):
        model.train()
        #print(f"Train pass {epoch}")
        print (f"Epoch: {epoch}")
        train_pass(model,train_dataloader, loss_fn, optimizer, device, train_loss_tracker, train_performance_tracker, regularization_in_loss)

        model.eval()
        validation_pass(model,validation_dataloader, loss_fn, device, validation_loss_tracker, validation_performance_tracker, regularization_in_loss)

        #print (f"Epoch: {epoch}")
        print (f"Training loss: {train_loss_tracker.batch_avg} \t Training perf metric {train_performance_tracker.batch_avg}")
        print (f"Validation loss: {validation_loss_tracker.batch_avg} \t Validation perf metric {validation_performance_tracker.batch_avg}")

        train_loss_tracker.advance_epoch()
        train_performance_tracker.advance_epoch()
        validation_loss_tracker.advance_epoch()
        validation_performance_tracker.advance_epoch()

        if early_stopping:
            early_stopping_handler(early_stopping_criterion,
                                    model, f"{checkpoint_location}/{checkpoint_name}", 
                                    train_loss_tracker,
                                    train_performance_tracker,
                                    validation_loss_tracker,
                                    validation_performance_tracker)

    return train_loss_tracker, train_performance_tracker, validation_loss_tracker, validation_performance_tracker

