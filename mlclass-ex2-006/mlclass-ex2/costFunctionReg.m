function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



z=X*theta;
htheta = sigmoid(z);
p1=sum(-(y.*log(htheta))-((1-y).*log(1-htheta)))/m;

theta2=sum(theta.^2);
theta2=theta2-(theta(1)*theta(1));


p2=theta2*lambda/(2*m);
J=p1+p2;


%=================

p = htheta-y;

n=size(theta);

for i=1:n
a = p.*X(:,i);

b=sum(a)/m;
if(i==1)
grad(i)=b;
else
grad(i)=b+lambda*theta(i)/m;
end



% =============================================================

end
