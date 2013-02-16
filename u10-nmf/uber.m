function uber
  tra_p = load('usps.ascii/train_patterns.txt'); 
  

  [W H] = nnmf(tra_p, 100);
  
  save('w_uber.mat','W');
  save('w_uber.mat','H');
end