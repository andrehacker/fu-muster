% trasform data to a new basis, given by eigenspace (eigenvectors)
function r = pca_transform(eigenspace, data, dim)
  r = (eigenspace(:,end-dim+1:end)'*data')';
end