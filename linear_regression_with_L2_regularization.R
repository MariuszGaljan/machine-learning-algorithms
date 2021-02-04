# Script with functions for linear regression with regularization


predict_y = function(X, theta) {
  # adding 1s to match the bias from theta
  X = cbind(rep(1, nrow(X)), X)
  return(X %*% theta)
}


# Cost is MSE
calculate_cost_L2_reg = function(X, y, theta, lambda = 0) {
  m = nrow(X)
  predictions = predict_y(X, theta)
  squared_errors = (predictions - y)^2
  regularization = lambda/(2*m) * sum((theta[-1])^2)
  return(sum(squared_errors) / (2*m) + regularization)
}


calculate_cost = function(X, y, theta) {
  return(calculate_cost_L2_reg(X, y, theta, lambda = 0))
}


perform_gradient_descent_L2_reg = function(X, y, theta, alpha=0.005, epsilon=10^-10, lambda=0, max_iterations=300000) {
  X_scaled = scale(X)
  X_bias = cbind(rep(1, nrow(X_scaled)), X_scaled)
  
  act_error = calculate_cost_L2_reg(X_scaled, y, theta, lambda)
  step_improvement = 1
  m = nrow(X)
  iterations = 0
  
  
  while (step_improvement > epsilon && iterations < max_iterations) {
    gradient = 1/m * t(X_bias) %*% (X_bias %*% theta - y)
    
    gradient_regularization = lambda/m * theta
    gradient_regularization[1] = 0
    gradient = gradient + gradient_regularization
    
    theta = theta - alpha * gradient
    
    previous_error = act_error
    act_error = calculate_cost_L2_reg(X_scaled, y, theta, lambda)
    
    step_improvement = abs(act_error - previous_error)
    iterations = iterations + 1
    
    if (act_error - previous_error > 0) {
      print('Cost is increasing. Alpha may be too big')
      return(list('theta' = theta, 'cost' = act_error))
    }
  }
  
  # Reverting standardization
  standard_theta = theta
  standard_theta_without_bias = standard_theta[2:nrow(standard_theta),]
  
  theta_without_bias = standard_theta_without_bias / apply(X, 2, sd)
  theta_bias = standard_theta[1, 1] - sum(standard_theta_without_bias * (apply(X, 2, mean) / apply(X, 2, sd)))
  theta = matrix(c(theta_bias, theta_without_bias))
  
  print(paste('Gradient descent completed after', iterations, 'iterations.'))
  print(paste('Trained cost: ', act_error))
  
  return(list('theta' = theta, 'cost' = act_error))
}
