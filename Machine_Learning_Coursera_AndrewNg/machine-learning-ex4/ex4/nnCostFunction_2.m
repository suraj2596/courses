function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a2 = [ones(m,1) sigmoid(X*(Theta1)')];
%size(a2)
a3 = sigmoid(a2*(Theta2)');
%size(a3)
%size(y)
ym = [y zeros(m, num_labels-1)];

for q=1:m
  z = zeros(1,num_labels);
  t = y(q,1);
  z(1,t) = 1;
  ym(q,:) = z;
end
size(y);

J = -((log(a3).*ym)+(log(1-a3).*(1-ym)));

J = sum(sum(J))/m;

%regularization
%(size(Theta1,1))
theta1_r = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
theta2_r = [zeros(size(Theta2,1),1) Theta2(:,2:end)];

r = (sum(sum(theta1_r.^2)) + sum(sum(theta2_r.^2)))*(lambda)/(2*m);
J = J+r;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
Delta2=zeros(size(theta2_r));
Delta1=zeros(size(theta1_r));
for i=1:m
  a2_bp = [1 sigmoid(X(1,:)*theta1_r')];
  size(a2_bp);
  a3_bp = [sigmoid(a2_bp*theta2_r')];
  %disp(ym(i,:))
  d3 = a3_bp - ym(i,:);
  d2 = (d3 * theta2_r(:, 2:end)).*sigmoidGradient(X(i,:)*theta1_r');
  Delta2 = (Delta2 + d3'*a2_bp);
  Delta1 = (Delta1 + d2'*X(i,:));
end


Theta2_grad = Delta2./m;
Theta1_grad = Delta1./m;


%size(a3_bp)
%size(ym)
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
