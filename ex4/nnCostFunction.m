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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m, 1) X];  % add bias

z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];  % add bias

z3 = a2 * Theta2';
a3 = sigmoid(z3);
% [max_element, max_index] = max(a3);
% h = max_index;  % a column vector giving predicted label for every training eg

h = a3;

yk = zeros(m, num_labels);

% the trick is to create a 2D matrix where each row specifies o/p of ...
% every neuron
for i=1:m,
  yk(i, y(i)) = 1;
end;


J = (-1/m)*(sum(sum(((yk .* log(h))  + ((1-yk) .* log(1-h))))));

% Note that you should not be regularizing the terms that correspond ...
% to the bias. For the matrices Theta1 and Theta2, this corresponds to 
% the first column of each matrix.

t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% regularization formula
reg_param = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);


J = J + reg_param;

% BACKPROP
% first forward propagate & calculate h

a1 = [ones(m, 1) X];  % add bias, m x 1+f 
z2 = a1 * Theta1';    % m x 1+f * 1+f x S 
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];  % add bias m x 1+S
z3 = a2 * Theta2';    % m x 1+S * 1+S x num_labels
a3 = sigmoid(z3);     % m x num_labels
% [max_element, max_index] = max(a3);
% h = max_index;  % a column vector giving predicted label for every training eg
h = a3;               % m x num_labels

% now start backprop & update the weights

delta_3 = h - yk;          % m x num_labels

% (S x num_labels) *  (num_labels x m)
delta_2 = (Theta2'(2:end, :) * delta_3') .* sigmoidGradient(z2'); 

% calculate weight updation
Theta1_grad += (1/m) * delta_2 * a1;   % S x m * m x 1+f
Theta2_grad += (1/m) * delta_3' * a2;  % num_labels x m * m x 1+S

% Theta1_grad is S x 1+f
% Theta2_grad is num_labels x 1+S

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ...
                            (lambda/m)*Theta1(:, 2:end); % S * 1+f
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ...
                            (lambda/m)*Theta2(:, 2:end);  % num_labels * 1+S


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
