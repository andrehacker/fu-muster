function run_nmf(iterations)
  tra_p = load('usps.ascii/train_patterns.txt'); 
  
  sizes = [40 60 80];
  Ws = cell(8,1);
  Hs = cell(8,1);

  parfor i = 1:3
    [W H] = nmf(tra_p, sizes(i), iterations);
    Ws{i}(:,:) = W;
    Hs{i}(:,:) = H;
  end
  
  for i = 1:3
    num = sprintf('%d',sizes(i));
    its = sprintf('%d',iterations);
    W = Ws{i}(:,:);
    H = Hs{i}(:,:);
    nw = ['w' its '_' num '.mat']
    nh = ['h' its '_' num '.mat']
    save(nw,'W');
    save(nh,'H');
  end  
end


