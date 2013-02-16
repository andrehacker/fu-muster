function compute_knn
  tra_p = load('usps.ascii/train_patterns.txt'); 
  tra_l = load('usps.ascii/train_labels.txt');
  tes_p = load('usps.ascii/test_patterns.txt');
  tes_l = load('usps.ascii/test_labels.txt');


  %[neighbors distances] = kNearestNeighbors(tra_p', tes_p', 3);
  %save('plain_neighbors3.mat','neighbors');

  % pca
  traX = tra_p';
  traY = dummyToNumber(tra_l');
  tesX = tes_p';
  tesY = dummyToNumber(tes_l');

  pcaBase = pca_get_base(traX);
  traX_t = pca_transform(pcaBase, traX, 100);  % Original has 256
  tesX_t = pca_transform(pcaBase, tesX, 100);  % Original has 256

  [neighbors distances] = kNearestNeighbors(traX_t, tesX_t, 3);
  save('pca_neighbors3.mat','neighbors');

  % nmf
  W = load('w_300_50.mat'); W = W.W;
  H = load('h_300_50.mat'); H = H.H;
  
  H_tes = ((W'*W)^-1)*W'*tes_p;
  [neighbors distances] = kNearestNeighbors(H', H_tes', 3);
  save('nmf_neighbors3.mat','neighbors');
end

function numbers = dummyToNumber(dummies)
  [value index] = max(dummies, [], 2);
  numbers = index - 1;
end