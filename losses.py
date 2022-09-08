
import torch.nn as nn
import torch as ts
import numpy as np
class L_G(nn.Module):
    def __init__(self, lamdba_val_1 , iters ):
        super(L_G,self).__init__()
        self.lambda_val_1 =  lamdba_val_1
        self.lambda_val_2 =  lambda x : 15/ iters *x + 1 # Function that denotes the linear function that descibes \lambda_{g_2}
    def forward(self, samples, truth, d, current_iter):
        sum = 0
        for x in range(samples.shape[0]):
            sum += self.lambda_val_1 + np.abs(d[x]) + self.lambda_val_2(current_iter) * ts.dist(samples[x] - truth[x])
        return 1 / samples.shape[0] * sum

class L_l(nn.Module):
    def __init__(self):
        super(L_G,self).__init__()
    def forward(self,label,distances,k,num_samples):
        # k muss leranable sein 
        sum = 0
        for x in range(num_samples):
            sum += nn.BCELoss(label[x],nn.Sigmoid(k*distances[x]))
        return 1 / num_samples * sum 

class L_e(nn.Module):
    def __init__(self):
        super(L_G,self).__init__()
    def forward(self,normals,num_samples):
        # k muss leranable sein 
        sum = 0
        for x in range(num_samples):
            sum += ( ts.norm(normals[x]) - 1 ) * ( ts.norm(normals[x]) - 1 )
        return 1 / num_samples * sum 
class L_a(nn.Module):
    def __init__(self,l_1,l_2):
        super(L_G,self).__init__()
        self.lambda_a_1 = l_1
        self.lambda_a_2 = l_2
    def forward(self,albedo_front,albedo_back,truth_front,truth_back,num_samples_f , num_samples_O):
        # k muss leranable sein 
        sum_1 = 0
        sum_2 = 0
        for x in range(num_samples_f):
            sum_1 += ts.dist(albedo_front[x],truth_front[x])
        for x in range(num_samples_O):
            sum_2 += ts.dist(albedo_back[x],truth_back[x])
        
        return 1/num_samples_f * self.lambda_a_1 * sum_1 + 1/num_samples_O * self.lambda_a_2 * sum_2
class L_r(nn.Module):
    def __init__(self):
        super(L_G,self).__init__()
    def forward(self,rays,albedo_front,albedo_back,truth_front,truth_back):
        sum = 0 
        for x in range(rays.shape[0]):
            sum += ts.dist(albedo_front[x],truth_front[x])
            sum += ts.dist(albedo_back[x],truth_back[x])
        
        return 1/rays.shape[0] * sum 

class L_c(nn.Module):
    def __init__(self):
        super(L_G,self).__init__()
    def forward(self,albedo_front,shading,pixels,rays_number):
        sum = 0
        for x in range(rays_number):
            sum += ts.abs(ts.mult(albedo_front[x],shading[x]) - pixels[x])
        return 1 / rays_number * sum
class L_s(nn.Module):
    def __init__(self):
        super(L_G,self).__init__()
    def forward(self,shading_net,normals,truth_front_albedo,image,l):
        image = ts.flatten(image)
        sum = 0
        for x in range(image.shape[0]):
            sum += ts.abs(ts.mult(truth_front_albedo[x],shading_net(normals[x],l)) - image[x])
        
        return 1 / image.shape[0] * sum 


def positional_encoding(x,y,z,l):
    for _ in range(l):
        new_x =  ts.tensor([np.sin((2^t)*x*np.pi) if t%2 else np.cos((2^t)*x*np.pi) for t in range(l*2) ])
        new_y =  ts.tensor([np.sin((2^t)*y*np.pi) if t%2 else np.cos((2^t)*y*np.pi) for t in range(l*2) ])
        new_z =  ts.tensor([np.sin((2^t)*z*np.pi) if t%2 else np.cos((2^t)*z*np.pi) for t in range(l*2) ])
        vector = ts.cat((new_x,new_y,new_z), dim = 0)
        return vector

    