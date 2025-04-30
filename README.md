# SOAP with update norm momentum

SOAP optimizer modification (original SOAP - https://github.com/nikhilvyas/SOAP/blob/main/soap.py). Updates are normalized to exponential moving average (EMA) of previous updates. For extra stability norm of the EMA is not allowed to grow by more than 1.1 times per step. 

Basically this is a very simple change but beats other optimizers that I have been testing over last few months on 8/10 of my benchmarks, and it can be applied to any other optimizer but I used SOAP because it performed the best in my benchmarks. And then it also cooked on my actual U-Net task (no lr sweep tho because it takes one day to train a model)

# Benchmarks
I have been testing various optimizers on a set of benchmarks and this has beaten all other optimizers on many of them so I decided to upload this. I do a learning rate sweep and plot the smoothed loss curve of the best run on the left, and learning rate to loss graph on the right. The loss curve plots are pretty useless IMO but the right side lr plots are quite informative. The graphs are quite a mess though, I do apologize for that and for tiny text.

Since I only have a laptop with GTX1650 I did most benchmarks on [MNIST1d](https://github.com/greydanus/mnist1d) to make it LR sweeps viable. Its a 1d version of MNIST, and it is also supposed to be much harder than MNIST. It has 4000 samples by default so there is often insane amount of overfitting, which sounded good for a benchmark when I was deciding what to put in it. Also because of overfitting train losses are sometimes plotted on log scale while test losses are plotted on linear scale. I also do 1000 to 3000 steps which is kinda bad but I can't increase it, it would just take too much time.

The plots also show the best optimizer so far in my testing, plus some reference optimizers - SGD, Adam, SOAP, Kron and Muon. Note that Muon maybe underperforms because it applies to last two dimensions instead of first two (I used this one https://github.com/KellerJordan/Muon) and I forgot to switch it.

Also all iamges below are available as one big 16 MB image https://github.com/inikishev/SOAP-with-update-norm-momentum/blob/main/summary%20-%20SOAP%20%2B%20Normalize%20to%20EMA.jpg. 

# ML
### mnist1d classification fullbatch 4 layer MLP 
This version beats everything else on train losses, by the way look at the test losses, the NGD optimizer is this one https://github.com/YiwenShaoStephen/NGD-SGD, I don't know why it worked so well there, it also works well on other tasks too
![image](https://github.com/user-attachments/assets/a1ed707f-7ff3-4f93-8c51-d0f8fd1ebe4a)

### mnist1d classification mini-batch 4 layer MLP
Also beats everything else on train losses, the "SOAP-NormalizeByEMA" is the same optimizer as in this repo, but from torchzero and the implementation might be slightly different, I think the main difference is that I set precondition_1d to True by default.
![image](https://github.com/user-attachments/assets/e40e66ed-d334-4976-873c-89a2db0df23d)

### mnis1d classification mini-batch 2 layer RNN
Also beats everything else on train losses. Also Kron seems to be very good at not overfitting on this and previous one.
![image](https://github.com/user-attachments/assets/975035f8-3709-4575-ae4e-8999a903fc3e)

### mnist1d classification mini-batch 4 layer ConvNet
The best optimizer on train losses here, "Orthogonalize-NAG" is Muon (without Adam) + nesterov momentum applied to orthogonalized gradients. And this one is from torchzero and applied to first two dims.
![image](https://github.com/user-attachments/assets/5c3b6294-d04e-4721-86dc-20809257886f)

### mnist1d autoencoding mini-batch 4 layer ConvNet
Again, "SOAP-NormalizeByEMA" is the same as this repo but implementation from torchzero 
![image](https://github.com/user-attachments/assets/ca408b65-9516-4cd8-a8a8-1e607f60c1c7)

### 1d segmentation on a custom synthetic dataset with dice focal loss - 3 layer ConvNet
Beats everything else on train losses. But here I think test losses actually mean that it has some overfitting.
![image](https://github.com/user-attachments/assets/24e450d9-93e2-4905-beeb-c528b41b733f)

# Synthetic
### 512-dimensional quadratic function
It's hard to beat L-BFGS although I haven't tested NewtonCG because I haven't made it yet, by the way for some reason L-BFGS works better when applying preconditioning to momentum and grafting to cautious Adam.
![image](https://github.com/user-attachments/assets/7999cfe7-eec5-48cf-9d80-1290e3ba11a1)

### Matrix inversion with L1 loss
I picked this and the following benchmarks because they seems to mirror ML benchmarks but it is much faster to evaluate. Goal in this one is to invert a matrix. MARS has been the leader here for a long time but this beats it finally.
![image](https://github.com/user-attachments/assets/9f40a219-9df5-442c-9a84-6fc8cb7d8d1f)

### Inverse matrix inversion with MSE loss
The goal here is to find a matrix whose inverse is the input matrix, so it is the same objective as previous one but formulated differently and involves gradients through torch.linalg.inv
![image](https://github.com/user-attachments/assets/2c65b005-d288-4a17-a1ab-7ddeb260aac9)

### Matrix sign
The goal is to minimize residual between two newton schulz step appled to an input matrix while also minimizing MSE with original matrix, basically its an amalgation of objectives but also mirrors ML performance well for some reason, while L-BFGS hs a slither of lrs with very good performance.
![image](https://github.com/user-attachments/assets/c3dd2c81-8e1d-46aa-964f-beec46f590c6)

# benchmarks code
it is tecnhically available in one of my repos but it is a mess but I can clean it up and upload here if needed
