% Parameter: lcavol lweight age lbph svi lcp gleason pgg45 lpsa train (F/T)

function linreg
  % =======
  % TASK 1
  % =======

  tra = dlmread('prostate-tra.mat', ' ');
  tes = dlmread('prostate-tes.mat', ' ');

  % Convenience
  traIn = tra(:,1:end-1); 
  tesIn = tes(:,1:end-1);
  traOut = tra(:, end);
  tesOut = tes(:, end);

  % Plot distribution of lpsa
  plotOutputDistribution(traOut, tesOut);

  % Add 1 dimension for offset/bias
  traIn = [ones(size(tra, 1),1) traIn];
  tesIn = [ones(size(tes, 1),1) tesIn];
 
  % 1) Compute weights minimizing the squared error
  weights = getWeightsLeastSquares(traIn, traOut);
  
  % 2) Compute squared error
  sumErrorTra = sumSqError(weights, traIn, traOut);
  sumErrorTes = sumSqError(weights, tesIn, tesOut);
  % Nice to have:
  meanErrorTra = meanSqError(weights, traIn, traOut);
  meanErrorTes = meanSqError(weights, tesIn, tesOut);
  mAbsDevTra = meanAbsDev(weights, traIn, traOut);
  mAbsDevTes = meanAbsDev(weights, tesIn, tesOut);

  %Write result to file
  fid = fopen('task1-results.txt','w');
  fprintf(fid, 'Fitting was always done based on Training data\n');
  fprintf(fid, 'Error-rate was determined for both, Training and Test data\n\n');
  fprintf(fid, 'Sum Squared Error (Training): %.3f\n', sumErrorTra);
  fprintf(fid, 'Sum Squared Error (Test): %.3f\n\n', sumErrorTes);
  fprintf(fid, 'Mean Squared Error (Training): %.3f\n', meanErrorTra);
  fprintf(fid, 'Mean Squared Error (Test): %.3f\n\n', meanErrorTes);
  fprintf(fid, 'Mean absolute Deviation (Training): %.3f\n', mAbsDevTra);
  fprintf(fid, 'Mean absolute Deviation (Test): %.3f\n', mAbsDevTes);
  fclose(fid);

  % =======
  % TASK 2
  % =======
  h1 = figure('Name','Sum Squared Errors for subsets (Training)','NumberTitle','off');
  xlabel('Size of subset (k)', 'FontSize', 17);
  ylabel('sum squared error', 'FontSize', 17);
  hold on;
  h2 = figure('Name','Mean squared Errors for subsets (Training)','NumberTitle','off');
  xlabel('Size of subset (k)', 'FontSize', 17);
  ylabel('mean squared error', 'FontSize', 17);
  hold on;
  h3 = figure('Name','Mean absolute Errors for subsets (Training)');
  xlabel('Size of subset (k)', 'FontSize', 17);
  ylabel('mean absolute error', 'FontSize', 17);
  hold on;
  h4 = figure('Name','Sum Squared Errors for subsets (Test)');
  xlabel('Size of subset (k)', 'FontSize', 17);
  ylabel('sum squared error', 'FontSize', 17);
  hold on;
  
  fid = fopen('task2-results.txt','w');
  for k=1:8
    comb = nchoosek(1:8,k);
    results = zeros(size(comb,1), 1);
    listweights = zeros(size(comb,1), k+1);
    for l=1:size(comb,1)
      % Reduce to current parameters
      samplesTra = tra(:,comb(l,:));
      samplesTes = tes(:,comb(l,:));
      samplesTra = [ones(size(samplesTra, 1),1) samplesTra];
      samplesTes = [ones(size(samplesTes, 1),1) samplesTes];
      % apply lin. reg.
      weights = getWeightsLeastSquares(samplesTra, traOut);
      % compute error-measures
      sumErrorTes = sumSqError(weights, samplesTes, tesOut);
      sumErrorTra = sumSqError(weights, samplesTra, traOut);
      meanErrorTes = meanSqError(weights, samplesTes, tesOut);
      meanAbsErrorTes = meanAbsDev(weights, samplesTes, tesOut);
      figure(h1);
      scatter(k, sumErrorTra, 200, 'g+');
      figure(h2);
      scatter(k, meanErrorTes, 200, 'g+');
      figure(h3);
      scatter(k, meanAbsErrorTes, 200, 'g+');
      figure(h4);
      scatter(k, sumErrorTes, 200, 'g+');

      % Store combination and error rate in cell array to sort
      results(l, 1) = sumErrorTes;
      listweights(l, :) = weights;
    end
    printResults(results, comb, listweights, fid);
  end
  fclose(fid);

  print(h1,'-dpng','task2-sum-sq-errors.png');
  print(h2,'-dpng','task2-mean-sq-errors.png');
  print(h3,'-dpng','task2-mean-abs-errors.png');
  print(h4,'-dpng','task2-sum-sq-errors-tes.png');

  % =======
  % ADDON 1
  % Plot lin. regression for single features
  % =======
  for i=1:8
    plotRegressionForFeature(tra, i);
  end
  
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
% input = list of row vectors
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
function s = getCovar(samples, mu)
  % Normalize and compute covar
  samples = samples - (ones(size(samples,1),1) * mu);
  s = samples' * samples;
  s = s / size(samples,1);

  % make some noise;)
  s = s + (0.0001 * eye(size(s)));
end

