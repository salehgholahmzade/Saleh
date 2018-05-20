
from NeuralModel.NeuralModel import *





################## Genrate the dat set #################################3333

data,label=generate_disc_set(2000)

train_input = data[0:1000].view(1000,2)
train_label = label[0:1000].long().view(1000,1)
test_input = data[1000:2000].view(1000,2)
test_label = label[1000:2000].long().view(1000,1)

#########################################################################33


############### Instantiate a model #################################3333
model = sequential(linear(2,25),Relu(),linear(25,25),Relu(),linear(25,25),Relu(),linear(25,25),Tanh(),linear(25,1),sigmoid())
####################################################################################################33


#################### Train model ###############################################333
list_loss = train_model(model,train_input, train_label, mini_batch_size=1,epoch=75,learning_rate=0.00001)

# plot training loss


plt.plot(list_loss[0:25],color='g' )
plt.title('loss_plot for training')
plt.xlabel('number of epochs')
plt.ylabel("loss")

###################################################################################3


#################### Make prediction from model ####################################33
print('shape of test input is  = ',test_input.shape) # shape of test set
#### run forward pass to predict for a model after training
print('Runing prediction for  model --------------#')
pred=list(np.zeros(1000))
for i in range(1000):
    pred[i] = model.forward(test_input[i].view(2,1))[-1]

print('shape of prediction output is -------------',len(pred)) 
###########################################3 shape of test set##########################################################################3333

####################### Compute error ############################################33
print('computing accuracy of model after prediction ------------------------------#')
err=0
pred = np.asarray(pred)
mask=pred>0.5
pred1 = mask.astype(int)
#print(pred1)
#print(test_label.numpy().reshape(1000))
err = np.sum(pred1!=test_label.numpy().reshape(1000))
achievable_ = 1;
accuracy = (achievable_ - err/pred1.shape) * 100
#print('The total label the model got wrong ---------', err)
print('accuracs of model is ---------', accuracy)
#################################################################################3 
