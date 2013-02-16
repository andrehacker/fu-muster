function test_results
  tra_p = load('usps.ascii/train_patterns.txt'); 
  tes_p = load('usps.ascii/test_patterns.txt');

  tra_l = load('usps.ascii/train_labels.txt');
  [val tra_l] = max(tra_l);
  tra_l = tra_l-1;

  tes_l = load('usps.ascii/test_labels.txt');
  [val tes_l] = max(tes_l);
  tes_l = tes_l-1;

  p = predict(tra_l, 'nmf_neighbors3.mat');
  isVsShould = [p' tes_l'];
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0));
  NmfSuccessRate = right /size(diff)

  p = predict(tra_l, 'plain_neighbors3.mat');
  isVsShould = [p' tes_l'];
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0));
  PlainSuccessRate = right /size(diff)

  p = predict(tra_l, 'pca_neighbors3.mat');
  isVsShould = [p' tes_l'];
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0));
  PcaSuccessRate = right /size(diff)

end


function r = predict(tra_l,filename)
  results = load(filename,'neighbors');
  results = results.neighbors;
  r = mode(tra_l(results)');
end