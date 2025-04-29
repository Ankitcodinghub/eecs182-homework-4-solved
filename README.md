# eecs182-homework-4-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Homework 4 Solved](https://www.ankitcodinghub.com/product/eecs182-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116343&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Homework 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1. Understanding Dropout (Coding Question)

In this question, you will analyze the effect of dropout in a simplified setting. Please follow the instructions in the Jupyter notebook and answer the questions in your submission of the written assignment. The notebook does not need to be submitted.

(a) (No dropout, least-square) The mathematical expression of the OLS solution, and the solution calculated in the code cell.

(b) (No dropout, gradient descent) The solution in the code cell. Are the weights obtained by training with gradient descent the same as those calculated using the closed-form least squares method (c) (Dropout, least-square) The solution in the code cell.

(d) (Dropout, gradient descent) Describe the shape of the training curve. Are the weights obtained by training with gradient descent the same as those calculated using the closed-form least squares method?

(e) (Dropout, gradient descent, large batch size) Describe the loss curve and compare it with the loss curve in the last part. Why are they different? Also compare the trained weights with the one calculated by the least-square formula.

(f) Refer back to the cells you ran in part (e). Analyze how and why adding dropout changes the following: (i) How large were the final weights w1 and w2 compared to each other. (ii) How large the contribution of each term (i.e. 10w1 + w2) is to the final output. Why does this change occur? (This does not need to be a formal math proof).

(g) (Optional) Sweeping over the dropout rate Fill out notebook section (G). You should see that as the dropout rate changes, w1 and w2 change smoothly, except for a discontinuity when dropout rates are 0. Explain this discontinuity.

(h) (Optional) Optimizing with Adam: Run the cells in part (H). Does the solution change when you switch from SGD to Adam? Why or why not?

(i) Dropout on real data: Run the notebook cells in part (I), and report on how they affect the final performance.

2. Regularization and Dropout

You saw one perspective on the implicit regularization of dropout in HW, and here, you will see another one. Recall that linear regression optimizes the following learning objective:

(1)

One way of using dropout during SGD on the d-dimensional input features xi involves keeping each feature at random ∼i.i.d Bernoulli(p) (and zeroing it out if not kept) and then performing a traditional SGD step.

It turns out that such dropout makes our learning objective effectively become

(2)

where ⊙ is the element-wise product and the random binary matrix R ∈ {0,1}n×d is such that Ri,j ∼i.i.d Bernoulli(p). We use wˇ to remind you that this is learned by dropout.

Recalling how Tikhonov-regularized (generalized ridge-regression) least-squares problems involve solving:

(3)

for some suitable matrix Γ.

(a) Show that we can manipulate (2) to eliminate the expectations and get:

(4)

with Γˇ being a diagonal matrix whose j-th diagonal entry is the norm of the j-th column of the training matrix X.

(b) How should we transform the wˇ we learn using (4) (i.e. with dropout) to get something that looks a solution to the traditionally regularized problem (3)?

(Hint: This is related to how we adjust weights learned using dropout training for using them at inference time. PyTorch by default does this adjustment during training itself, but here, we are doing dropout slightly differently with no adjustments during training.)

(c) With the understanding that the Γ in (3) is an invertible matrix, change variables in (3) to make the problem look like classical ridge regression:

(5)

Explicitly, what is the changed data matrix Xe in terms of the original data matrix X and Γ?

(d) Continuing the previous part, with the further understanding that Γ is a diagonal invertible matrix with the j-th diagonal entry proportional to the norm of the j-th column in X, what can you say about the norms of the columns of the effective training matrix Xe and speculate briefly on the relationship between dropout and batch-normalization.

3. Weights and Gradients in a CNN

In this homework assignment, we aim to accomplish two objectives. Firstly, we seek to comprehend that the weights of a CNN are a weighted average of the images in the dataset. This understanding is crucial in answering a commonly asked question: does a CNN memorize images during the training process? Additionally, we will analyze the impact of spatial weight sharing in convolution layers. Secondly, we aim to gain an understanding of the behavior of max-pooling and avg-pooling in backpropagation. By accomplishing these objectives, we will enhance our knowledge of CNNs and their functioning.

Let’s consider a convolution layer with input matrix X ∈ Rn×n,

  x1,1 x1,2 ··· x1,n

x2,1 x2,2 ··· x2,n

 , (6)

X =  … … … … 

  xn,1 xn,2 ··· xn,n

weight matrix w ∈ Rk×k,

  w1,1 w1,2 ··· w1,k

w2,1 w2,2 ··· w2,k

 

w =  … … … … , (7)

  wk,1 wk,2 ··· wk,k

and output matrix Y ∈ Rm×m,

  y1,1 y1,2 ··· y1,m

y2,1 y2,2 ··· y2,m 

Y =  … … … … . (8)

  ym,1 ym,2 ··· ym,m

For simplicity, we assume the number of the input channel (of X is) and the number of the output channel (of output Y) are both 1, and the convolutional layer has no padding and a stride of 1. Then for all i,j,

k k

yi,j = XXxi+h−1,j+l−1wh,l, (9)

h=1 l=1

or

Y = X ∗ w, (10)

, where ∗ refers to the convolution operation. For simplicity, we omitted the bias term in this question.

Suppose the final loss is L, and the upstream gradient is dY ∈ Rm,m,



dy1,1

dy2,1 dY =  …





dym,1 dy1,2 dy2,2

…

dym,2 ···

··· …

··· 

dy1,m

dy2,m  

… ,



dym,m (11)

where dyi,j denotes .

(a) Derive the gradient to the weight matrix dw ∈ Rk,k,

  dw1,1 dw1,2 ··· dw1,k

dw2,1 dw2,2 ··· dw2,k

dw =  … … … … , (12)

  dwk,1 dwk,2 ··· dwk,k

where dwh,l denotes . Also, derive the weight after one SGD step with a batch of a single

image.

(b) The objective of this part is to investigate the effect of spatial weight sharing in convolution layers on the behavior of gradient norms with respect to changes in image size.

For simplicity of analysis, we assume xi,j,dyi,j are independent random variables, where for all i,j:

E[xi,j] = 0, (13)

Var(xi,j) = σx2, (14)

E[dyi,j] = 0, (15)

Var(dyi,j) = σg2. (16)

Derive the mean and variance of for each i,j a function of n,k,σx,σg. What is

the asymptotic growth rate of the standard deviation of the gradient on dwh,l with respect to the length and width of the image n?

Hint: there should be no m in your solution because m can be derived from n and k.

Hint: you cannot assume that xi,j and dyi,j follow normal distributions in your derivation or proof.

(c) For a network with only 2×2 max-pooling layers (no convolution layers, no activations), what will be ? For a network with only 2×2 average-pooling layers (no convolution

layers, no activations), what will be dX?

HINT: Start with the simplest case first, where X . Further assume that top left value is

selected by the max operation. i.e.

y1,1 = x1,1 = max(x1,1,×1,2,x2,1,×2,2) (17)

Then generalize to higher dimension and arbitrary max positions.

(d) Following the previous part, discuss the advantages of max pooling and average pooling in your own words.

4. Inductive Bias of CNNs (Coding Question)

In this problem, you will follow the EdgeDetection.ipynb notebook to understand the inductive bias of CNNs.

(a) Overfitting Models to Small Dataset: Fill out notebook section (Q1).

(i) Can you find any interesting patterns in the learned filters?

(b) Sweeping the Number of Training Images: Fill out notebook section (Q2).

(i) Compare the learned kernels, untrained kernels, and edge-detector kernels. What do you observe?

(ii) We freeze the convolutional layer and train only final layer (classifier). In a high data regime, the performance of CNN initialized with edge detectors is worse than CNN initialized with random weights. Why do you think this happens?

(c) Checking the Training Procedures: Fill out notebook section (Q3).

(i) List every epochs that you trained the model. Final accuracy of CNN should be at least 90% for 20 images per class.

(ii) Check the learned kernels. What do you observe?

(iii) (optional) You might find that with the high number of epochs, validation loss of MLP is increasing whild validation accuracy increasing. How can we interpret this?

(iv) (optional) Do hyperparameter tuning. And list the best hyperparameter setting that you found and report the final accuracy of CNN and MLP.

(v) How much more data is needed for MLP to get a competitive performance with CNN? Does MLP really generalize or memorize?

(d) Domain Shift between Training and Validation Set: Fill out notebook section (Q4).

(i) Why do you think the confusion matrix looks like this? Why CNN misclassifies the images with edge to the images with no edge? Why MLP misclassifies the images with vertical edge to the images with horizontal edge and vice versa? (Hint: Visualize some of the images in the training and validation set.)

(ii) Why do you think MLP fails to learn the task while CNN can learn the task? (Hint: Think about the model architecture.)

(e) When CNN is Worse than MLP: Fill out notebook section (Q5).

(i) What do you observe? What is the reason that CNN is worse than MLP? (Hint: Think about the model architecture.)

(ii) Assuming we are increasing kernel size of CNN. Does the validation accuracy increase or decrease? Why?

(iii) How do the learned kernels look like? Explain why.

(f) Increasing the Number of Classes: Fill out notebook section (Q6).

(i) Compare the performance of CNN with max pooling and average pooling. What are the advantages of each pooling method?

5. Memory considerations when using GPUs (Coding Question)

In this homework, you will run GPUMemory.ipynb to train a ResNet model on CIFAR-10 using PyTorch and explore its implications on GPU memory.

We will explore various systems considerations, such as the effect of batch size on memory usage and how different optimizers (SGD, SGD with momentum, Adam) vary in their memory requirements.

(a) Managing GPU memory for training neural networks (Notebook Section 1).

(i) How many trainable parameters does ResNet-152 have? What is the estimated size of the model in MB?

(ii) Which GPU are you using? How much total memory does it have?

(iii) After you load the model into memory, what is the memory overhead (MB) of the CUDA context loaded with the model?

(b) Optimizer memory usage (Notebook Section 2).

(i) What is the total memory utilization during training with SGD, SGD with momentum and Adam optimizers? Report in MB individually for each optimizer.

(ii) Which optimizer consumes the most memory? Why?

(c) Batch size, learning rates and memory utilization (Notebook Section 3)

(i) What is the memory utilization for different batch sizes (4, 16, 64, 256)? What is the largest batch size you were able to train?

(ii) Which batch size gave you the highest accuracy at the end of 10 epochs?

(iii) Which batch size completed 10 epochs the fastest (least wall clock time)? Why?

(iv) Attach your training accuracy vs wall time plots with your written submission.

6. Homework Process and Study Group

We also want to understand what resources you find helpful and how much time homework is taking, so we can change things in the future if possible.

(a) What sources (if any) did you use as you worked through the homework?

(b) If you worked with someone on this homework, who did you work with?

List names and student ID’s. (In case of homework party, you can also just describe the group.)

(c) Roughly how many total hours did you work on this homework? Write it down here where you’ll need to remember it for the self-grade form.

.
