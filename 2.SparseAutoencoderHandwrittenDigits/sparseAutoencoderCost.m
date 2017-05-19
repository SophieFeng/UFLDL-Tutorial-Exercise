function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

[FeaDimen,PatchNum] = size(data);

% 计算前向网络
l2z = W1 * data + repmat(b1,1,PatchNum);
l2a = sigmoid(l2z);
l3z = W2 * l2a + repmat(b2,1,PatchNum);
l3a = sigmoid(l3z);

%计算神经元的平均活跃度
activeP = sum(l2a,2)./PatchNum;

% 计算反向传播
l3delt = -(data-l3a) .* (l3a .* (1-l3a));
sparsity_penalty = beta.*(-(sparsityParam./activeP) + ((1-sparsityParam)./(1-activeP)));
l2delt = (W2'*l3delt + repmat(sparsity_penalty,1,PatchNum)) .* (l2a .* (1-l2a));

W1grad = l2delt * data';  
b1grad = sum(l2delt,2);
W2grad = l3delt * l2a';
b2grad = sum(l3delt,2);

co = sum((l3a - data).^2) / 2;
co = sum(co,2);
sparsityPart = sparsityParam*log(sparsityParam./activeP) + (1-sparsityParam).*log((1-sparsityParam)./(1-activeP));
cost = co/PatchNum + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2))) + beta * sum(sparsityPart);

%fprintf('cost = %d\n',cost);

%更新权重参数
W1grad = W1grad / PatchNum + lambda*W1;
b1grad = b1grad / PatchNum;
W2grad = W2grad / PatchNum + lambda*W2;
b2grad = b2grad / PatchNum;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

