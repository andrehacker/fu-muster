function solution()

  % TASK 1

  % Train Codebook & Encoding

  tra_p = load('usps.ascii/train_patterns.txt'); 
  tra_l = load('usps.ascii/train_labels.txt');
  
  %[W H] = nmf(tra_p, 30, 500);
  %save('w_500_30.mat','W');
  %save('h_500_30.mat','H');
  W = load('w_500_30.mat'); W = W.W;
  H = load('h_500_30.mat'); H = H.H;
  %SizeW = size(W)
  %SizeH = size(H)

  if 1==0
    % Visualize original input
    train_patterns_small = load('usps.ascii/train_patterns_small');
    a_few_img = train_patterns_small.train_patterns_small;
    SizeFewImages = size(a_few_img)
    visualize_simple(a_few_img,'original');

    % Visualize codebook
    visualize_simple(W,'codebook');

    % Visualize reproduced digits (Training data)
    Reproduced = W*H;
    visualize_simple(Reproduced(:,1:8), 'reproduced');
  end

  % -------
  % TASK 2
  % -------
  tes_p = load('usps.ascii/test_patterns.txt');
  tes_l = load('usps.ascii/test_labels.txt');
  traX = tra_p';
  traY = dummyToNumber(tra_l');
  tesX = tes_p';
  tesY = dummyToNumber(tes_l');

  % a) No Preprocessing
  disp('Start classification (no preprocessing)');
  Ntes = 5; %size(tesX, 1);
  isVsShould = zeros(Ntes, 2);
  parfor i = 1:Ntes
    isVsShould(i,:) = ...
      [linearKNNSearch(2, traX, traY, tesX(i,:)), tesY(i)];
  end
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0))
  SuccessRate = right / size(diff)

  % b) Preprocessing with PCA
  disp('Start classification (pca)');
  pcaBase = pca_get_base(traX);
  traX_t = pca_transform(pcaBase, traX, 1);  % Original has 256
  tesX_t = pca_transform(pcaBase, tesX, 1);  % Original has 256
  isVsShould = zeros(Ntes, 2);
  parfor i = 1:Ntes
    isVsShould(i,:) = ...
      [linearKNNSearch(2, traX_t, traY, tesX_t(i,:)), tesY(i)];
  end
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0))
  SuccessRate = right / size(diff)
  return
  
  % c) Preprocessing with NMF
  disp('Start classification (nmf)');
  disp('TODO');

end

function numbers = dummyToNumber(dummies)
  [value index] = max(dummies, [], 2);
  numbers = index - 1;
end


function visualize_nmf_codebook(W,filename)
  h = figure; 
  subplots = [];
  for i = 1:size(W,2)
    small_img = reshape(W(:,i),16,16)';
    subplots = [subplots subplot(ceil(size(W,2)/2),2,i)]; 
    imagesc(small_img);
    %axis([1 16 1 16]);
    colormap(gray);
  end
  axis(subplots, 'square');
  print(h,'-deps',[filename '.eps']);
end


function visualize_simple(W,filename)
  for i = 1:size(W,2)
    h = figure; 
    small_img = reshape(W(:,i),16,16)';
    imagesc(small_img);
    colormap(gray);
    num = sprintf('%d',i);
    print(h,'-deps',[filename num  '.eps']);
  end
end