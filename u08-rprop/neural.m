% TODO: Define error-threshhold for termination of learning
% TODO: Handle local minima: Start with multiple random weights. For each, do maxIteration iterations and choose best

function neural()

  % Task 2 - Learn Pendigits
  task1();
end

function task1()

  % If weights are bad, more iterations don't help us.
  % => Try often with few iterations, take best

  % -------
  % TASK 1 Compare Online and Offline
  % -------
  penTra = load('pendigits.tra');
  penTes = load('pendigits.tes');
  ion = load('ionosphere.data');  %34 features, 351 items, feature 2 always 0
  [ionTra ionTes] = randomPartition(ion, 0.7);
  penTraX = penTra(:,1:end-1);
  penTesX = penTes(:,1:end-1);
  ionTraX = ionTra(:,1:end-1);
  ionTesX = ionTes(:,1:end-1);

  % Normalize to deviation from std-dev
  penTraX = standardize(penTraX);
  penTesX = standardize(penTesX);
  ionTraX = standardize(ionTraX);
  ionTesX = standardize(ionTesX);

  % Convert output to dummy variable
  penTraY = toDummy(penTra(:,end));
  penTesY = toDummy(penTes(:,end));
  ionTraY = toDummy(ionTra(:,end));
  ionTesY = toDummy(ionTes(:,end));

  % PCA (does not improve results)
  %pcs = principalComponents(penTraX);
  %penTraX = transformData(pcs, penTraX, 14);
  %penTesX = transformData(pcs, penTesX, 14);

  % PEN Online
  weightsFrom = -1;  % Interval for random weights
  weightsTo = 1;
  learnRate = 0.1;  % Learning rate
  itemsPerIteration = 1;
  hidden = 20;  % Hidden nodes
  penOnlineIts = 10.^[0:5];
  sPenOnline = runTests(penOnlineIts, penTraX, penTraY, penTesX, penTesY, ...
      hidden, learnRate, itemsPerIteration, weightsFrom, weightsTo)

  % Ion Online
  learnRate = 0.1;  % Learning rate
  itemsPerIteration = 1;
  hidden = 40;  % Hidden nodes
  ionOnlineIts = 10.^[0:5];
  sIonOnline = runTests(ionOnlineIts, ionTraX, ionTraY, ionTesX, ionTesY, ...
      hidden, learnRate, itemsPerIteration, weightsFrom, weightsTo)

  % PLOT
  h = figure();
  semilogx(ionOnlineIts, sIonOnline', '-xr', 'Displayname', 'Ion success rate');
  hold on;
  semilogx(penOnlineIts, sPenOnline', '-+r', 'Displayname', 'Pen success rate');
  xlabel('Iterations', 'FontSize', 15);
  ylabel('Success Rate', 'FontSize', 15);
  legend('show');
  legend('boxoff');
  print(h,'-deps',['online' '.eps']);

  return;

  % learnRate = 0.001;  % Good learning rate for offline

  %EabsMean = Eabs./Ntra;
  %EsqMean = Esq./Ntra;
  %plotErrorForIterations([([0:size(Eabs,1)-1]*errEvery)' EabsMean], ...
  %    'task2-abs-err', 'Iterations','Mean Absolute Error');
  %plotErrorForIterations([([0:size(Esq,1)-1]*errEvery)' EsqMean], ...
  %    'task2-sq-err', 'Iterations','Half squared Error');
  %plotErrorForIterations([([0:size(Esq,1)-1]*errEvery)' sum(EsqMean,2)/M ], ...
  %    'task2-sq-mean-err', 'Iterations','Mean half squared Error');
  %plotErrorForIterations([([0:size(Eonline,1)-1])' Eonline ], ...
  %    'task2-sq-mean-err-online', 'Iterations','Mean half squared online Error');

end

% A row vector with all number of iterations to execute
function s = runTests(its, TraX, TraY, TesX, TesY, ...
      hidden, learnRate, itemsPerIteration, weightsFrom, weightsTo)
  fid = fopen('task1-results.txt','a');
  fprintf(fid, '\nStart Testrun\n');
  s=[];
  for i=1:size(its,2)
    maxIterations = its(i);
    errEvery = maxIterations;

    tic()
    [W1d W2d Eabs Esq Eonline] = learnNeural(TraX, TraY, hidden, size(TraY,2), ...
        learnRate, maxIterations, errEvery, ...
        itemsPerIteration, weightsFrom, weightsTo);
    elapsed = mytoc('Time for learning pen (seconds): ');
    s = [s; neuralTestSuccess(W1d, W2d, TesX, TesY)];

    fprintf(fid, 'Pen Its: %.0f\tItemsPerIt: %.0f\tLearnRate: %.4f\tHidden: %.0f\tSeconds: %.2f\n', 10^i, ...
      itemsPerIteration, learnRate, hidden, elapsed);
    fprintf(fid, '> Success Rate: %.2f\n', s(i,1));
  end
  fclose(fid);
end

% Splits data into training and test set
% traPercentage: From 0 to 1
function [tra tes] = randomPartition(data, traPercentage)
  N = size(data,1);
  traIds = randsample(N, round(N*traPercentage));
  tra = data(traIds, :);
  tesIds = setdiff([1:N],traIds);
  tes = data(tesIds, :);
end

% Classify input and compute success rate
% W1, W2 = weights with bias
% X = input, without bias
% Y = rows of dummy variables with result
% sr = success rate in percentages
function sr = neuralTestSuccess(W1d, W2d, X, Y)
  Xd = [X, ones(size(X,1), 1)];
  isVsShould = zeros(size(X,1),2);
  for i = 1:size(X,1)
    [value realClass] = max(Y(i,:));
    realClass = realClass - 1;
    predictedClass = classifyNeural(W1d, W2d, Xd(i,:));
    isVsShould(i,:) = [predictedClass realClass];
  end
  diff = isVsShould(:,1) - isVsShould(:,2);
  right = size(diff(diff==0));
  sr = right / size(diff) * 100;
end

% Returns a 1xM vector
function r = evalNetwork(W1d, W2d, xd)
  o1 = sigmoid(xd*W1d);   % 1xK
  o1d = [o1 1];
  r = sigmoid(o1d*W2d);  % 1xM
end

function c = classifyNeural(W1d, W2d, xd)
  dummyOut = evalNetwork(W1d, W2d, xd);
  [value index] = max(dummyOut);
  c = index-1;
end

% Input: Nx1 vector with scalar values (classes)
%   Assumption: Classes are integers (Class 1 = value 1)
% Output: NxC vector with dummy variable in each row
function R = toDummy(Y)
  numberClasses = max(Y);
  R = zeros(size(Y,1), numberClasses);
  for i=1:size(Y,1)
    R(i, Y(i)+1) = 1;
  end
end

% Learns the weights of a 2 Layer Neural Network
% Uses Online (sequential) Training
% @param X: Samples
% @param Y: Output (encoded as dummy variables)
% @param K: Number of hidden nodes
% @param M: Number of output nodes
% @param learnRate: Learning rate
% @param maxIterations: when to stop
% @param errEvery: Error (for all items) will be calculated only every ... iterations.
%                  Error computation is very costly, so we don't want to do it in every step
% @return W1d: weights of the 1st layer
% @return W2d: weights of the 2nd layer
% @return Eabs: Matrix. Eabs(i,j) is the absolute avarage deviation
%   (abs(predicted - y)) for the j-th output unit in iteration i. For
%   all samples
% @return Esq: Matrix. Esq(i,j) is the half squared avarage deviation
%   (((predicted - y)^2)/2) for the j-th output unit in iteration i. For
%   all samples
% @return Eonline: column vector. Eonline(i) is the the half
%   squared error for the current random sample in iteration i of
%   online learning. That is the function that we optimize for.
function [W1d W2d Eabs Esq Eonline] = learnNeural(X, Y, ...
      K, M, learnRate, maxIterations, errEvery, ...
      itemsPerIteration, weightsFrom, weightsTo)

  D = size(X, 2);   % number of input nodes (dimensionality of input)
  N = size(X, 1);   % number of input items
  Xd = [X, ones(size(X, 1), 1)];  % convenience

  Eabs = zeros(ceil(maxIterations/errEvery), M); % Track global (for all estraining items) absolute deviation for each output unit
  Esq = zeros(ceil(maxIterations/errEvery), M);
  Eonline = zeros(maxIterations, 1);

  % Start with random initial weights
  % 1st col = all weights to 1st node
  W1 = rand(D, K)*(weightsTo-weightsFrom) + weightsFrom;
  W2 = rand(K, M)*(weightsTo-weightsFrom) + weightsFrom;
  W1d = [W1; rand(1, K)*(weightsTo-weightsFrom) + weightsFrom];
  W2d = [W2; rand(1, M)*(weightsTo-weightsFrom) + weightsFrom];

  for i=1:maxIterations
    % Error-Tracking (optional)
    if (mod(i-1, errEvery) == 0)
      tmp=zeros(N, M);
      for l=1:N
        tmp(l,:) = evalNetwork(W1d, W2d, Xd(l,:));
      end
      Eabs((i-1)/errEvery+1, :) = sum(abs(tmp - Y), 1);
      Esq((i-1)/errEvery+1, :) = sum(((tmp - Y).^2),1) / 2;
    end

    % 0) Choose inputs for this iteration
    if itemsPerIteration > 1
      if itemsPerIteration >= N
        % Offline Learning
        XSub = X;
        XdSub = X;
        YSub = Y;
      else
        % Offline Learning, but only for subset
        % We could always do this, but it's slows down other cases
        ids = randsample(N,itemsPerIteration);
        XSub = X(ids, :);
        XdSub = Xd(ids, :);
        YSub = Y(ids, :);
      end
    else
      % 0) Online learning: Pick one random input
      id = randi(size(X,1));
      XSub = X(id, :);
      YSub = Y(id, :);
    end


    DeltaW1d = zeros(D+1, K);
    DeltaW2d = zeros(K+1, M);
    for j=1:itemsPerIteration % For offline learning

      x = XSub(j, :);
      xd = [x 1];
      y = YSub(j, :);

      % 1) Forward Propagation:
      % 1.1) Compute Layer hidden layer output o1 and Output layer output o2
      o1 = sigmoid(xd*W1d);   % 1xK
      o1d = [o1 1];
      o2 = sigmoid(o1d*W2d);  % 1xM
      o2d = [o2 1];

      % 1.2) Compute Derivations for Hidden units (D1) and Output Units (D2) (diagonal matrix)
      D1 = diag(o1.*(1-o1));   % KxK
      D2 = diag(o2.*(1-o2));   % MxM

      % 1.3) Compute Derivation DE for Error unit
      DE = (o2-y)';  % Mx1
                    %Eonline(i) = sum((DE.^2)/2)/M;  % TODO: Do this only once for Online!

      % 2) Back Propagation
      % to Output Layer
      backErr2 = D2*DE;
      % to Hidden Layerm
      backErr1 = D1 * W2 * backErr2;
      partialDerivativesW2 = (backErr2 * o1d)';
      partialDerivativesW1 = (backErr1 * xd)';


      % 3) Compute weight adoptions
      DeltaW2d = DeltaW2d + (-learnRate * backErr2 * o1d)';
      DeltaW1d = DeltaW1d + (-learnRate * backErr1 * xd)';
    end
    % 4) Update adoptions
    W2d = W2d + DeltaW2d;
    W1d = W1d + DeltaW1d;
    W2 = W2d(1:end-1,:);
    W1 = W1d(1:end-1,:);
  end
end

% Logistic sigmoid function
function r = sigmoid(x)
  r = 1./(1 + exp(-x));
end

% Standardize data by subtracting the mean and deviding by the standard
% deviation
function r = standardize(samples)
  m = mean(samples);
  sd = std(samples);
  % Attention: sd may never be 0!
  sd(sd==0)=0.0000001;
  for i = 1:size(samples,1)
    samples(i,:) = (samples(i,:) - m) ./ sd;
  end
  r = samples;
end

% trasform data to a new basis, given by eigenspace (eigenvectors)
function r = transformData(eigenspace, data, dim)
  r = (eigenspace(:,end-dim+1:end)'*data')';
end

% computes the principal components for the given data
% r = eigenvectors of the covariance matrix
function r = principalComponents(data)
  covarMatrix = cov(data);
  [r eigen_values] = eig(covarMatrix);
end

% Plot Error rate over iterations
function plotErrorForIterations(E, name, xtitle, ytitle)
  h = figure();
  hold on;
  xlabel(xtitle, 'FontSize', 15);
  ylabel(ytitle, 'FontSize', 15);
  for i=2:(size(E,2))
    plot(E(:,1),E(:,i));
  end
  print(h,'-deps',[name '.eps']);
end


function elapsed = mytoc(timerDescription)
  global total;
  %disp( ceil(toc()) )
  elapsed = toc();
  fprintf( '%s %.6f\n', timerDescription, elapsed );
  %printf([timerDescription mat2str(ceil(toc()))]);
end
