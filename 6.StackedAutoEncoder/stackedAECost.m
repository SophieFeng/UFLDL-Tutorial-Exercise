function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example

%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.


%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.

% 前向传递
l1a = data;
l2z = stack{1}.w * l1a + repmat(stack{1}.b,1,M);
l2a = sigmoid(l2z);
l3z = stack{2}.w * l2a + repmat(stack{2}.b,1,M);
l3a = sigmoid(l3z);

% softmaxd 概率向量
thetaX = softmaxTheta * l3a;
thetaX = bsxfun(@minus, thetaX, max(thetaX,[],1));
ethetaX = exp(thetaX);
p = bsxfun(@rdivide,ethetaX,sum(ethetaX));

% 计算梯度
delta3 = -(softmaxTheta' * (groundTruth-p)) .* (l3a .* (1-l3a));
delta2 = (stack{2}.w' * delta3) .* (l2a .* (1-l2a));

stackgrad{1}.w = delta2 * data'/M;
stackgrad{1}.b = sum(delta2,2)/M;
stackgrad{2}.w = delta3 * l2a'/M;
stackgrad{2}.b = sum(delta3,2)/M;

softmaxThetaGrad = -1/M *  (groundTruth - p) * l3a' + lambda * softmaxTheta;

%计算代价
sparse_hypo = log(p) .* groundTruth;
decay = lambda/2 * sum(sum(softmaxTheta.^2));
cost = -1/M * sum(sum(sparse_hypo)) + decay;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
