import torch.nn as nn
import torch
from models.regularization.regularization_calc import RegularizationDiag

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=1e-3):
        super(DiagonalLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False, device=device)
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight.data,a=0,b=1)
            #self.linear.weight.data.fill_(0.5)
        self.mask = torch.eye(in_features, dtype=bool, device=device)
        self.threshold = threshold

    def forward(self, x):
        self.linear.weight.data *= self.mask
        over_threshold = torch.abs(self.linear.weight.data) > self.threshold
        self.linear.weight.data *= over_threshold
        return self.linear(x)


class RegMlpModel(nn.Module):
    N = 10000000000000

    def __init__(self, input_shape, num_classes, threshold):
        super(RegMlpModel, self).__init__()
        self.threshold=threshold
        self.one_to_one = DiagonalLinear(input_shape, input_shape, threshold=threshold)
        self.regularization_calc = RegularizationDiag(
            torch.diagonal(self.one_to_one.linear.weight)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(input_shape, 2 * input_shape),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2 * input_shape),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2 * input_shape, int(0.5 * input_shape)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(int(0.5 * input_shape)),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(int(0.5 * input_shape), num_classes),
        )
        self.avg = torch.mean(torch.diagonal(self.one_to_one.linear.weight))
        self.discount = 0.9999
        self.lamda = 1

    def regularization(self):
        non_zero = torch.nonzero(torch.diagonal(self.one_to_one.linear.weight))
        ones = torch.diagonal(self.one_to_one.linear.weight)>0.9
        zeros = torch.diagonal(self.one_to_one.linear.weight)<0.09
        print("ones",sum(ones),"zeros",sum(zeros))
        #regu = self.regularization_l1()
        #regu = self.regularization_l1_until_target(50)
        regu = self.regularization_sigma_abs()
        regu_target = self.regularization_target(50)
        print("non zeros", len(non_zero), "regu", regu, "regu_target", regu_target)
        return regu


    def regularization_l1(self):
        return torch.norm(torch.diagonal(self.one_to_one.linear.weight), 1)

    def regularization_sigma_abs(self):
        x = torch.diagonal(self.one_to_one.linear.weight)
        res = torch.norm(x*(1-x), 1)
        return res

    def regularization_target(self,target):
        x = torch.diagonal(self.one_to_one.linear.weight)
        res = (torch.norm(x, 1)-target) ** 2
        return res

    def regularization_ex(self,target):
        non_zero = torch.nonzero(self.one_to_one.linear.weight)
        x=torch.diagonal(self.one_to_one.linear.weight)
        res=torch.pow(x,2)
        res=torch.pow(1-torch.exp(-1*res),2)
        res=torch.pow(res, 0.0001)
        return (torch.nansum(res)-target)**2

    def regularization_limit_bands(self, bands_goal):
        # scale numbers to be 0 it close to one, or 1 else
        eps = 1e-7
        t = torch.diagonal(self.one_to_one.linear.weight)
        t = torch.pow(t, 2)
        res = torch.log(t + eps)
        res = torch.pow(res, 2)
        res2 = torch.sum(torch.pow(res, 1 / RegMlpModel.N))
        return torch.sqrt(torch.pow(res2 - bands_goal, 2))

    def regularization_l1_until_target(self, target):
        non_zero = torch.nonzero(torch.diagonal(self.one_to_one.linear.weight))
        if len(non_zero) > target:
            return self.regularization_l1()
        return 0


    def regularization_l1_special(self, target):
        return torch.sqrt(torch.pow(self.regularization_l1() - target * self.avg, 2))

    def regularization_binary_exist(self, target):
        res = torch.diagonal(self.one_to_one.linear.weight) > 0.1
        res2 = torch.where(res == True)
        return (len(res2) - target) ** 2

    def forward(self, x):
        x = self.one_to_one(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
