import os
import openslide
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import math
from PIL import Image
from sklearn.model_selection import train_test_split
import Random_Rotation_Mirroring as RRM ##mine

result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"
a = np.load(result_addrs + "GrayScale_pixels.npy")
np.savetxt(result_addrs+'aaaa.csv', a, delimiter=',', fmt='%s')
print a
np.sort(a, axis=-1, kind='quicksort', order=None)


""" Build and train a convolutional neural network with TensorFlow."""
#input data
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"
file = result_addrs + "_info_gathering.npy"
info_gathering = np.load(file)
num_of_samples = sum(info_gathering[:,10])


# Training Parameters
patch_size = tf.constant(info_gathering[0][9]) #???
feasure_size = tf.constant(patch_size**2)
times = tf.constant(100) # equal to x times more than ...
learning_rate = tf.constant(1/(feasure_size)) #???
num_steps = tf.constant(feasure_size) #???
display_step = tf.constant(100) #???
batch_size = tf.constant(math.ceil(num_of_samples/feasure_size*times)) #???
ttt_ratio = 0.33 #train to test ration
sel_patches = tf.random_uniform(np.empty[0,math.ceil(num_of_samples*ttt_ratio)], minval=0, maxval=num_of_samples, dtype=tf.int32)
#normalize the data!!!
#making train set
for b in range(0,batch_size):
    for f in size(0,feasure_size*times):
        sel_patches_each_batch = tf.contrib.framework.sort(sel_patches[f*feasure_size*times,f*(feasure_size*times+1)],axis=1,direction='ASCENDING')
        
    cnt = 0
    j = 0
    train_set_ = list()
    for i in range(0,(feasure_size)*times):  #???  
        if i < cnt + info_gathering[j]:
            k = 0
            for file in sorted(os.listdir(result_addrs)):
                if file.sratrswith(result_addrs + "patch_list") and file.endswith(".ndpi"):                
                    k +=1  
                    
            for t in range(0,k):        
                if i < cnt+len(file):
                    file = result_addrs + "patch_list" + str(k) + ".ndpi"
                    loaded_mat = np.load(file)
                    tmp = loaded_mat[i-cnt,:,:,:]            
                    train_set.append(RRM.random_rot_mirr(tmp[i-cnt,:,:,:])) 
        else:
            j = j+1        
"""It id done!"""
#save the ID of the train patches, to find the test set

# Network Parameters
num_input = 10000 #data input (img shape: 100*100)
num_classes = 2 #total classes (Normal, Cancerous)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))

























































#const = tf.constant(10)
#fill_mat = tf.fill((4,4),10)
#myzeros = tf.zeros((4,4))
#myones = tf.ones((4,4))
#myrandn = tf.random_normal((4,4), mean=0, stddev=1.0)
#myrandu = tf.random_uniform((4,4), minval=0, maxval=1)
#my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]   
#sess = tf.InteractiveSession()
#for op in my_ops:
    #print(sess.run(op))
    #print(op.eval()) 
    #print('\n')
    
#a = tf.constant(10)
#b = tf.constant(20)
#result = tf.matmul(a,b)
#sess.run(result)

#graph_one = th.Graph()
#graph_one = tf.get_default_graph()
#graph_two = tf.Graph()
#with graph_two.as_default():
    #print(graph_two is tf.get_default_graph())

#print(graph_two is tf.get_default_graph())

#sees = tf.InteractiveSession()
#my_tensor = tf.random_uniform((4,4),0,1)
#my_var = tf.Variable(initial_value=my_tensor)
#init = tf.global_variables_initializer()
#sess.run(init)
#sess.run(my_var)

#ph = tf.placeholder(tf.float32, shape=(None,5))
#np.random.seed(101)
#tf.set_random_seed(101)
#rand_a = np.random.uniform(0,100,(5,5))
#rand_b = np.randomunform(0,100,(5,1))
#a = tf.placeholder(tf.float32)
#b = tf.placeholder(tf.float32)
#add_op = a+b
#mul_op = a*b

#with tf.Session() as sess:
    #add_result = ses.run(add_op, feed_dict={a:rand_a,b:rand_b})
    #print(add_result)
    
    #mult_result = sess.run(mul_op, fedd_dict={a:rand_a, b :rand_b})

##Neural Network    
#n_features = 10
#n_dense_neurons = 3
#x = tf.placeholder(tf.float32,(None, n_features))
#W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
#b = tf.Variable(tf.ones([n_dense_neurons]))
#xW = tf.matmul(x,W)
#z = tf.add(xW,b)
#a = tf.sigmoid(z)
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
    #sess.run(init)
    #layer_out = sess.run(a, feed_dict={x:np.random.random([1,n])})
    
##Regression 
#x_data = np.linspace(0,10,10)+ np.randpm.uniform(-1.5,1.5,10)
#y_label = np.linspace(0,10,10)+ np.randpm.uniform(-1.5,1.5,10)
#plot.plot(x_data, y_label, '*')


#m = tf.Variable(np.random.rand(1))
#b = tf.Variable(np.random.rand(1))
#error = 0
#for x,y in zip(x_data, y_label):
    #y_hat = m*x+b
    #error += (y-y_hat)**2

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#train = optimizer.minimize(error)
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
    #sess.run(init)
    #training_steps = 1
    #for i in range(training_steps):
        #sess.run(train)

#final_slope, final_intercept = sess.run([m,b])
#x_data = np.linspace(-1,11,10)
#y_pred_plot = final_slope*x_test+final_intercept
#plot.plot(x_data, y_pred_plot, 'r')
#plot.plot(x_data, y_label, '*')

#x_data = np.linspace(0,10,1000000)
#noise = np.random.rand(len(x_data))
#y_true = (0.5*x_data)+5+noise
#x+df = pd.DataFrame(data=x_data,columns=['X Data'])
#y_df = pd.DataFrame(data=y_true,columns=['Y'])
#x_df.head()
#my_data = pd.concat([x_df,y_df].axis=1)
#my_data.head()
#my_data.sample(n=250).plot(kind='scatter', x='X Data',y ='Y')

#batch_size = 8
#m = tf.Variable(np.random.rand(1))
#b = tf.Varisble(np.random.rand(1))

#xph = tf.placeholder(tf.float32,[batch_size])
#yph = tf.placeholder(tf.float32,[batch_size])

#y_model = m*xph+b
#error = tf.reduce_sum(tf.suare(yph-y_model))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#train = optimizer.minimize(error)

#init = tf.global_variables_initializer()
#with tf.Session() as sess:
    #sess.run(init)
    #batches = 1000
    #for i in range(batches):
        #rand_ind = np.randpm.ranint(len(x_data),size = batch_size)
        #feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        #sess.run(train, feed_dict = feed)
        
#model_m, model_b = sess.run([m,b]) 
#y_hat = x_data*model_m+model_b
#my_data.samples(250).plot(kind='scater', x='X Data', y = 'Y')
#plt.plot(x_data,y_hat,'r')

#feat_cols = [tf.feature_column.numeric_column('x',shape =[1])]
#estimator =  tf.estimator.LinearRegressor(feature_columns = feat_cols)
#x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)
#input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8, num_epochs=None,shuff=True)
#train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8, num_epochs=1000,shuff=False)
#eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8, num_epochs=1000,shuff=False)
#estimator.train(input_fn=input_func,steps=1000)
#train_matrics = estimator.evaluate(input_fn=train_input_func, steps = 1000)
#eval_metrics = estimator.avaluate(input_fn=eval_func,steps=1000)
#brand_new_data = np.linspace(0,10,10)
#input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data}, shuffle = False)
#list(estimator.predict(input_fn=inpot_fn_predict)) # list the data
#predictions = []
#for pred in estimator.predict(input_fn = input_fn_predict):
    #predictions.append(pred['predictions'])
    
#my_data.sample(n=250).plot(kind='scatter',x='X Data', y = 'Y')
#plt.plot(brand_new_data, predictions, 'r*')

#diabetes = pd.read_scv('pima-indians-diabets.csv')
#diabets.head()
#cols_to_norm = ['BMI', 'Pedigree']
#diabests[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x:(x-x.min())/(x.max()-x.min()))
#num_preg = tf.feature_column.numeric_column('BMI')
#assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
##assigned_group = tf.feature_column.categorical_column_with_vhash_bucket('Group',hash_bucket_size=10)
#diabetes['Age'].hist(bins=20)
#feat_cols = [num_preg, insulin]
#labes = diabetes['Class']
#input_func = tf.estimator.inputs.pandas_input_fn(x=X-train, y= y_train, batch_size=10,num_epochs=1000,shuffle = True)
#model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
#model.train(input_fn=input_func,steps=1000)
#eval_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test, batch_size=10,num_epochs=1,shuffle = False)
#results = model.evaluate(eval_input_func)
#pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X-test,Batch_size=10,num_epochs=1, shuffle=False)
#predictions = model.predict(pred_input_func)
#my_pred = list(predictions)
#dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes = 2)
#dnn_model.train(input_fn= input_func, steps=1000)
#embedded_group_col= tf.feature_column.embedding_column(assigned_group,dimension= 2)
#feat_cols = [num_preg, insulin, embedded_group_col] 
#input_func= tf.estimstor.inputs.pandas_input_fn(X-train,Y_train,Batch_zise=10,num_epochs=1000, shuffle=True)
#dnn_model= tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols)
#dnn_model.train(input_fn=input_func,steps=1000)
#eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X-test,y=y_test, batch_size=10, num_epochs=1, shuffle = False)
#dnn_model.evaluate(eval_input_func)
#final_slope, final_intercept = sess.run([m,b])
#saver.save(sess,'models/my_first_model.ckpt') #6,39
#with tf.Session() as sess:
    #saver.restore(sess,'models/my_first_model.ckpt') #restore the model
    #restored_slope, restores_intercept = sess.run([m,b]) #fetch back results
    
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#mnist.train.images
#mnist.train.num_examples
#mnist.test.num_examples
#single_image = mnist.rain.images[1].reshape(28,28)


##placeholders
#x = tf.placeholder(tf.float32,shape=[None,784])
##variables
#W= tf.Variables(tf.zeros([784,10]))
#b = tf.Variables(tf.zeros[10])
##create graph operations
#y = tf.matmul(x,W)+b 
##loss function
##placeholder
#y_true = tf.placeholder(tf.float32,[None,10])
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
##optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
#train = optimizer.minimize(cross_entropy)
##create session
#init = tf.global_variables_initializer()
#with tf.Session as sess:
    #sess.run(init)
    #for step in range(1000):
        #batch_x, batch_y = mnist.train.next_batch(100)
        #sess.run(train,feed_dic={x:batch_x,y_true:batch_y})
        
##evaluate the model
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
#acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print(sess.run(acc,feed_dict={x:mnist.test.images,y-true:mnist.test.labels}))

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##helper
##init weights
#def init_weights(shape):
    #init_random_dist= tf.truncated_normal(shape,stddev=0.1)
    #return tf.Variable(init_bias_vals)
##conv2d
#def conv2d(x,W):
    #return tf.nn.con2d(x,W,strides=[1,1,1,1], padding='SAME')
##pooling
#def max_pooling_2by2(x):
    #return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

##convolutional layer
#def convolutional_layer(input_x,shape):
    #W = init_weights(shape)
    #b= init_bias([shape[3]])
    #return tf.nn.relu(conv2d(input_x,W)+b)

##normal (fully connected)
#def normal_full_layer(inpit_layer,size):
    #input_size = int(input_layer.get_shape()[1])
    #W = init_weights([input_size,size])
    #b = init_bias([size])
    #return tf.matmul(input_layer,W)+b