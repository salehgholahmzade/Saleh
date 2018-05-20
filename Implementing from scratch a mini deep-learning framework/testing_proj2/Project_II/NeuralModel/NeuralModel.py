

#################### BAsic libraries to import #######################################################################33
import torch
import numpy as np
import math
import matplotlib.pyplot as plt




########################################### Linear Modulrê ####################################################################3
class linear():
    '''
    Linear Function 

    Input : input size and output size
    Output : model parameters

    '''
    def __init__(self,start_layer_size,end_layer_size):
        assert type(start_layer_size)== int 
        assert type(end_layer_size) == int
        if (start_layer_size > 0 and end_layer_size > 0):
            
            self.m = start_layer_size
            self.n = end_layer_size
            self.w = torch.Tensor(self.n,self.m).normal_()
            self.b = torch.Tensor(self.n,1).normal_()
        else :
            raise "error"
        
    def forward(self, x):
        '''
        forward pass for linea

        '''
        #print('linear forward called')
        return torch.mm(self.w , x)+self.b
    
    def backward(self,x):
        '''
         BAckeward pass for linear 
        '''
        return x,self.w
    
    def parameter(self):
        '''
        Track model parameters

        '''
        return self.w,self.b
    
        
##############################################^ENd of linear Module ######################################################333




############################### Relu module #######################################################################3
class Relu():

    '''
    Activation Function Relu
    '''
     
    def forward(self,x):
        '''

        Forward pass for Relu
        '''
        y = torch.zeros_like(x)
        return torch.max(x,y)
    
    def backward(self,x):

        '''
        Backward computation for Relu


        '''
        u=x.clone()
        u[u>0]=1
        u[u<0]=0
        return u
    def parameters(self):
        return ["empty"]

################################### End of relu module ###########################################################################


################################ Tanh module ##############################################################################33

class Tanh():

    '''
   Tanh activation function definition


    '''
     
    def forward(self,x):
        '''
        Forward pass for for Relu

        '''
        return torch.tanh(x)
       
    
    def backward(self,x):
        return torch.ones_like(x)-torch.tanh(x)*torch.tanh(x)
    def parameters(self):
        return ["empty"]


######################################## End of Tanh module #######################################################################33

########################## Sigmoid Function ###############################################3
class sigmoid():
    '''

    Sigmoid function

    Applied the output of model
    to generate predictions
    between 0 and 1

    '''
     
    def forward(self,x):
        return torch.sigmoid(x)
       
    
    def backward(self,x):
        return torch.sigmoid(x)*(torch.ones_like(x)-torch.sigmoid(x))
    def parameters(self):
        return ["empty"]




########################333 Identity module ###############################################3###############################33
class Identity():
    '''
   The class has 3 functions each explanied below

   backward:

   backward function. This function returns identity 
   matrix if we have linear units following each other.
   If a linear unit is followed by activstion then we multiply the 
   unit matrix by derivative of the activation unitA

   parameters:

   It can also be called on modules that have no parameters in which case
   it returns empty list.


   forward:

   It returns the same input as the output if it is used in the forward pass
   computation. 


    '''
    
    def forward(self,x):
        '''
        compute the forward pass of a module
        '''
        return x
    
    def backward(self,x):
        '''
        compute the backward pass. But in this case it is a 
        matrix of ones. It is an helper function. It is inserted 
        inbetween linear units or activations and can be replaced 
        with the right actviation derivative in the case of a linear
        unit being followed by non linear unit.
        '''
        return torch.ones_like(x)
    def parameters(self):
        '''
         modules like relu that have no parameters uses this to return an empty list


        '''
        return ["empty"]
################################################## End of Identity module ###################################################################3

####################################### Helper Functions here ####################################################333
def __structrize__(operations):

      '''
         This is an internal function thats helps to keep track of the order
         in which a user specify the operations. Inparticular it gets the operation 
         name and stires it either as linear operation or âctivation .


        If a linear unit is followed by a nonlinear unit then we insert an identity in bet
        ween these modules. The identity operation use its backward operation
        that returns a matrix of one to add unit matrix inbetween a linear and non linear unit.

        This linear unit will be replaced by the backward pass for the non linear unit


      '''
      que=[] # store all operatiosn here
      linear_que=[] # list for lineâr unit
      activations_que = [] # list for activation
      l=len(operations) # length of operations
       #print(l)
      for i in range(l):
           que.append(operations[i])
           
           if(type(operations[i]).__name__=='linear'): # get  linear unit and store in linear_linque, a list
               linear_que.append(operations[i])
           
           
           if(i<l-1):
               if(type(operations[i]).__name__=='linear' and type(operations[i+1]).__name__=='linear'):
                   que.append(Identity())
       
      if (type(operations[-1]).__name__=='linear'): # append identity to last uni if linear
           que.append(Identity())
           
           
      activations_que = list(set(que).difference(linear_que)) # get the activation unit 
                                                            # we use the difference between the all operations
                                                            # and the linear unit that was initially extracted in linear_que above
       
      return que , linear_que , activations_que


############################## End of helpêr function #################################################################################3


######################## Sequential class main module ###################################################################################3

class sequential(): #list of operation
    '''
    This is the main module. It takes in the list of operation 
    and initialises the weights and biases using the linear ĉlass.
     
     We use structurize to get the linear unit and non linear unit in alist
     And add a unit matrix on beteen linear and non linear unit. Thus we
     generate a list with linear unit, a list with non linear unit and a list
     containing all operations with unit matrix inserted whenever a linear 
     unit is followed by a  non linea r unit

    It has the forwward and backward operation. This function defaults 
    to the forward and backward function for either the linear or non 
    linear unit. Basically it seperateŝ the linear unit from non linear unit 
    and applies the correct activation funtion based on the type of operation
    encountered in the list of operations.

    It also contains the MSE loss function.

    And also it has the zero:grad function that reinitialises gradients to zero
    for each iteration in a loop

    The update parameters update the parameters accordingly



    '''
    def __init__(self,*operations):
        
        #self.l = len(self.linear_operations)
        self.result = []
        self.linear_result=[]
        self.activation_result=[]
        
        self.linear_operations = []
        self.activation_operations=[]
        
        self.initial_operations = operations
        self.operations , self.linear_operations, self.activation_operations = __structrize__(self.initial_operations )
        
        self.l = len(self.linear_operations)
        
        self.pairs=[]
        self.mseloss=[]
      
        self.grad_w=[]
        self.grad_b=[]

        #         linear.__init__(self,in_node,out_node)
#         function.__init__(self)
        self.delta =[] # dl_ds
        self.forward_flag=False
        self.backward_flag=False

        self.dl_dw = []
        self.dl_db = [] 

    def loss(self ,target):
        if(self.forward_flag==False):
            raise ValueError("forward hasn't been called.")
        self.mseloss.append(torch.sum(torch.pow(self.result[-1]-target,2)))
        return self.mseloss
    
    
    def forward(self,x):

        '''

        does he forward computation of modules
        Initialises the self.forward flag to true- so that 
        when the backward pass is called no assertion error is raised

        The operation depends on the list of opearations parsed by caller


        '''
        self.forward_flag=True
        self.result.append(x)
        self.activation_result.append(x)
        for op in self.operations:
            
            tmp_result=(op.forward(self.result[-1]))
            self.result.append(tmp_result)
            
            if(type(op).__name__=='linear'):  
                
                self.linear_result.append(tmp_result)  # we store it for backward propagation 
            else:
                self.activation_result.append(tmp_result) # we store it for backward propagation
                


        return self.result 
        
    def backward_pass(self,target):
        '''
        BAckward pass that depends on the list
        of operations instantiated by the caller.

        It asserts that the forward module is called before
        using the backward module.


        '''
        
        if(self.forward_flag==False):
            raise ValueError("forward hasn't been called.")
            
        self.backward_flag=True
        self.dl_dw=[]
        self.dl_db=[]
        self.delta=[]
        
        self.delta.append(-2*(target-self.result[-1]))
        
        #print("result",len(self.result))
        
        
        for i in range (self.l,0,-1): # calculating deltas
       
            self.delta.append(torch.mm(torch.t(self.linear_operations[i-1].backward(self.linear_result[i-1])[1]),self.delta[-1])) # [1] is to select w not x from linear_backward()
       
        self.delta.reverse()
        for i in range (self.l,0,-1): #calculating dl_dw's 
           
            self.dl_dw.append(torch.mm(self.delta[i],torch.t(self.activation_result[i-1])))
        
        self.dl_dw.reverse()
        
    
        self.dl_db=self.delta[1:]
        return self.delta, self.dl_dw, self.dl_db

    def parameters(self):
        '''
        update the parameters of a module

        This module returns the pairs of parameters for each module and
        empty list of module has no parameter


        '''
        self.pairs=[]

        if(self.backward_flag==False):
                raise ValueError("backward hasn't been called.")
                
              
                
        for i, ops in enumerate(self.linear_operations):
           
            self.pairs.append([ops.w,self.dl_dw[i]] ) # appending w's and dl_dw's
            self.grad_w.append(self.dl_dw[i])
          
            self.pairs.append([ops.b,self.dl_db[i]] ) # appending b's and dl_db's
            self.grad_b.append(self.dl_db[i])
            
            self.pairs.append(self.operations[2*i+1].parameters()) # appending 'empty' for activation functions
          
            
                    
        return  self.pairs
    
    
    
    def zero_grad(self):
        '''

        initialiises all parameters to zero at each iteration


        '''
        
        self.grad_w=[]
        self.grad_b=[]
          
        self.dl_dw = []
        self.dl_db = [] 
        self.delta = []       
        self.result = []
        self.linear_result=[]
        self.activation_result=[]
        return self
    
    def update(self,eta =0.01):
        '''

        update the paramters 

        This is used in the training function to update parametwes of model


        '''
        self.result = []
        self.linear_result=[]
        self.activation_result=[]
        for i in range(self.l):
             
            #print('updating w[',i,"]")

            self.linear_operations[i].w = -eta * self.grad_w[i] + self.linear_operations[i].w
            #print('updating b[',i,"]")

            self.linear_operations[i].b = -eta* self.grad_b[i] + self.linear_operations[i].b
        #print("model has updated")


############################################3 End of sequential module ##############################################################3


############################################# Generate data set for testing and training module ##########################33

def generate_disc_set(nb):

    '''
    Generate data set to test model


    '''
    input_data = torch.Tensor(nb, 2).uniform_(-0.5,0.5)
    target = input_data.pow(2).sum(1).sub(1 /(2* math.pi)).sign().mul(-1).add(1).div(2).long()
    return input_data, target


################################# End of generate dataset ###################################################################333



############################33 Train model #############################################################################3


def train_model(model, train_input, train_target, mini_batch_size=1,epoch=25,learning_rate=0.01):

    '''
    This function helps train a model. 
    The losses are computed for each batch and stored in a list

    We call forward and backeward pass on the model.
    The model which alsready has the list of operations  applies the 
    forward and backward pass based on the type of operatio

    We compute the loss using loss function from sequetial and 
    store each batch loss in a list

    '''
    train_loss=[]
    for i in range(epoch):
        
        sum_loss = 0
    
        for b in range(0, train_input.size(0), mini_batch_size):
            #print(train_input.narrow(0, b, mini_batch_size))
            output = model.forward(torch.t(train_input.narrow(0, b, mini_batch_size)))
           
        
            model.backward_pass(train_target.narrow(0, b, mini_batch_size).type(torch.FloatTensor))
            #print("added_loss = ",model.loss(train_label.narrow(0, b, mini_batch_size).type(torch.FloatTensor))[-1])
            sum_loss = sum_loss + model.loss(train_target.narrow(0, b, mini_batch_size).type(torch.FloatTensor))[-1]/(train_input.size(0))    
            model.parameters()
            model.update(eta=learning_rate)
            model.zero_grad()    
          
        train_loss.append(sum_loss)
        print("epoch:",i," ,  loss= ", sum_loss)
   
    return train_loss


