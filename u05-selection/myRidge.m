% Parameter: lcavol lweight age lbph svi lcp gleason pgg45 lpsa train (F/T)

function myRidge
  % =======
  % TASK 1
  % =======
 
  tra = standardize(dlmread('prostate-tra.mat', ' '));
  tes = standardize(dlmread('prostate-tes.mat', ' '));
  trates = standardize(dlmread('prostate-all.mat', ' '));

  % Convenience
  traIn = tra(:,1:end-1);
  tesIn = tes(:,1:end-1);
  tratesIn = trates(:,1:end-1);
  traOut = tra(:, end);
  tesOut = tes(:, end);
  tratesOut = trates(:, end);
 
  % Add 1 dimension for offset/bias
  traIn = [ones(size(tra, 1),1) traIn];
  tesIn = [ones(size(tes, 1),1) tesIn];
  tratesIn = [ones(size(trates, 1),1) tratesIn];
 
  % 1) Ridgeregression
  labs = [];
  labs = [labs ; cellstr(['lcavol'])];
  labs = [labs ; cellstr(['lweight'])];
  labs = [labs ; cellstr(['age'])];
  labs = [labs ; cellstr(['lbph'])];
  labs = [labs ; cellstr(['svi'])];
  labs = [labs ; cellstr(['lcp'])];
  labs = [labs ; cellstr(['gleason'])];
  labs = [labs ; cellstr(['pgg45'])];
 
  plotRidgeRegression(tratesOut, trates(:,1:end-1),labs, 'ridgeregression-all');
  plotRidgeRegression(traOut, tra(:,1:end-1),labs,'ridgeregression-tra');
 
  % =======
  % TASK 2
  % =======
 
  % read the ALL of the data
  all = dlmread('prostate-all.mat', ' ');
 
  % we consider only features 1, 5 and 7
  testn = all(1:10,[1 5 7]);
 
  % Add 1 dimension for offset/bias
  testn = [ones(size(testn, 1),1) testn];
 
  q = 50;   % number of sample picks per iteration
  bootstraps = [];
  for i = 1:100
    % pick 50 random indecies
    ran = randi([1,size(all,1)],q,1);
    % get the samples at the random indecies
    allIn  = all(ran,[1 5 7]);
    % get the actual values at the indecies
    allOut = all(ran,9);
    % Add 1 dimension for offset/bias
    allIn = [ones(size(allIn, 1),1) allIn];
    % compute linear regression and add to bootstrap list
    bootstraps = [bootstraps getWeightsLeastSquares(allIn, allOut)];
  end
 
  [m d] = computeConfidence(bootstraps, testn);
  fid = fopen('task2-results.txt','w');
  fprintf(fid, 'Mean and 2*standard deviation for the first %d samples\n', ...
          size(testn,1));
  for i = 1:size(testn,1)
    fprintf(fid, 'Sample %d\t| Mean = %f | 2 x standard deviation =  %f\n', i, m(i,:),d(i,:));
  end
 
  % =======
  % TASK 3
  % =======
 
  eight = load('pendigits8.txt');
  eight = eight(:,1:end-1);
  [m covarianceMatrix] = meanAndCov(eight);
  covarianceMatrix = covarianceMatrix/norm(covarianceMatrix);
  ran = randi([1,100],16,1)';
  ran = ran/norm(ran);
  for i = 1:10
    ran = ran * covarianceMatrix;
    ran = ran/norm(ran);
  end
  [eigenvectors, eigenvalues] = eig(covarianceMatrix);
  pc =eigenvectors(:,end)';
  fid = fopen('task3-results.txt','w');
  fprintf(fid, 'experiment. result \t| longest eigenvector\n');
  fprintf(fid, '-------------------------------------\n');
  for i = 1:size(ran,2)
    fprintf(fid, '%+.3f\t\t|\t%+.3f\n',ran(:,i),pc(:,i));
  end

end

function r = foldDigit(digit)
  r = [];
  for i = 1:2:16
    r = [r ; digit(:,i:i+1)];
  end
end

% Computes mean and covariance based on matrix with observations
% Input: rows with training data for one class
% Output: Mean (row vector) and covariance matrix (16x16)
function [m c] = meanAndCov(data)
  m = mean(data);
  c = cov(data);
  c = c + (0.0001 * eye(size(c)));
end

% use bootstraped data for prediction
function [m d] = computeConfidence(bootstraps, y)
  m = [];
  d = [];
  for i = 1:size(y,1)
    p = [];
    for j = 1:size(bootstraps,2)
      p = [p predict(bootstraps(:,j), y(i,:))];
    end
    m = [m ; mean(p)];
    d = [d ; 2*std(p)];
  end
end

% Standardize data by subtracting the mean and deviding by the standard
% deviation
function r = standardize(samples)
  m = mean(samples);
  sd = std(samples);
  for i = 1:size(samples,1)
    samples(i,:) = (samples(i,:) - m) ./ sd;
  end
  r = samples;
end

% Computes the ridge regression parameters
function a = ridgeRegression(y, samples, lambda)
  a = [];
  for i = 1:size(lambda,2)
    t = inv(samples'*samples + (lambda(:,i)*eye(size(samples'*samples))));
    a = [a (t * samples'*y)];
  end
end

% df(l) function from Hastie
function r = df(samples, lambda)
  r = [];
  for i = 1:size(lambda,2)
    t = inv(samples'*samples + (lambda(:,i)*eye(size(samples'*samples))));
    a = samples * t * samples';
    r = [r trace(a)];
  end
end

% plot the ridge regression
function r = plotRidgeRegression(y, samples, labels, filename)
  k = [0:1:1000 30000];
  data = ridgeRegression(y,samples,k);
  plt = figure();
  xlim([-1 8]);
  ylim([-0.3 0.7])
  hold on;
  plot(df(samples,k), data,'LineWidth',2,'Color','blue');
  for i = 1:size(labels,1)
    text(size(data,1)+0.1, data(i,1) ,labels(i,:),'FontSize',10);
  end
  line([-1 8],[0 0], 'Color', 'black')
  xlabel('df(\lambda)', 'FontSize', 17);
  ylabel('Coefficients', 'FontSize', 17);
  print(plt,'-depsc',[filename '.eps']);
end

% Least Squares Fitting based on samples and output column vector
function w = getWeightsLeastSquares(samples, y)
  % Compute pseudo-inverse matrix
  pseudo = getPseudoInverse(samples);
  w = pseudo * y;
end

% Get pseudo-inverse matrix, needed for least squares fitting
function p = getPseudoInverse(X)
  % very likely that inverse exists if we have many samples
  p = inv(X' * X) * X';
end

% Predicts value for input based on weights
% samples = list of row vectors
% weights = column-vector
function r = predict(weights, samples)
  r = samples * weights;
end

% Return Sum of squared errors
function e = sumSqError(weights, samples, y)
  deviation = y - predict(weights, samples);
  e = deviation' * deviation;
end

% Get Mean Squared Error MSE
function e = meanSqError(weights, samples, y)
  e = sumSqError(weights, samples, y) / size(samples,1);
end

% Mean absolute deviations/errors (MAD, LAE)
% Though we did not use least absolute deviations method...
function e = meanAbsDev(weights, samples, y)
  e = sum(abs(y - predict(weights, samples))) ...
   / size(samples,1);
end

% Print results of subset task
function printResults(results, combs, weights, fid)
  % sort by error rate
  k = size(combs,2);
  [results, I] = sortrows(results);
  combs = combs(I,:);
  weights = weights(I,:);

  fprintf(fid, '\nk=%d\n', k);
  for i=1:size(results,1)
    fprintf(fid,'%s\t%.3f\t%s\n', ...
      mat2str(combs(i,:),0), ...
      results(i,1), ...
      mat2str(weights(i,:),3));
  end
end

% Plot correlation of one feature with output.
function plotRegressionForFeature(tra, feature)
  X = [ones(size(tra, 1),1) tra(:,feature)];
  weights = getWeightsLeastSquares(X, tra(:, end));
  Y = predict(weights, X);
  h10 = figure();
  hold on;
  xlabel('input x (value of feature)', 'FontSize', 17);
  ylabel('output (predicted vs real value)', 'FontSize', 17);
  scatter(X(:,2), tra(:,end), 'bo', 'filled', 'Displayname', 'Real lpsa');
  scatter(X(:,2), Y, 300, 'r*', 'Displayname', 'Predicted lpsa');
  title(['Feature ' mat2str(feature)], 'FontSize', 30);
  print(h10,'-dpng',['task2-feature' mat2str(feature) '.png']);
end

function plotOutputDistribution(traOut, tesOut)
  h = figure();
  hold on;
  xlabel('lpsa value', 'FontSize', 15);
  ylabel('propability density function PDF', 'FontSize', 15);

  % Just look at both, train and test:
  out = [traOut ; tesOut];

  %Plot values
  scatter(out, zeros(size(out,1),1), 100, 'bo', 'filled');

  %plot probability density function
  mu = getMean(out)
  sigma = getCovar(out, mu);
  ix = mu-3*sigma:0.01:mu+3*sigma;
  iy = gaussDensity(mu, sigma, ix')';
  plot(ix,iy);
  text(mu, gaussDensity(mu, sigma, mu) ,['Mean = ' mat2str(mu,4)], 'FontSize', 15);

  print(h,'-dpng',['output-distribution-tes.png']);
end

% compute prob. density for normal distribution
function p = gaussDensity(mu, sigma, data)
  normalize = 1 / (sqrt(2*pi) * sigma);
  p = zeros(size(data,1), 1);
  for i=1:size(data,1)
    p(i) = normalize * exp( -(data(i)-mu)^2 / (sigma^2) );
  end
end

% Computes mean based on matrix with observations
% Input: rows with training data for one class
% Output: Mean (row vector)
function m = getMean(data)
  m = 1/size(data,1) * sum(data);
end


% Compute Covariance matrix
function s =              getCovar(samples, mu)
  % Normalize and compute covar
  samples = samples - (ones(size(samples,1),1) * mu);
  s = samples' * samples;
  s = s / size(samples,1);

  % make some noise;)
  s = s + (0.0001 * eye(size(s)));
end
