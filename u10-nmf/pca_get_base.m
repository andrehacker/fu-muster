% computes the principal components for the given data
% r = eigenvectors of the covariance matrix
function r = pca_get_base(data)
  covarMatrix = cov(data);
  [r eigen_values] = eig(covarMatrix);
end