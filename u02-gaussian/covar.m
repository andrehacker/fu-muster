
% --------------
% Classifier using multivariant normal distribution
% --------------
function covar
  % Load input into (nx17) Matrix.
  % last column stores the class number
  tra = load('pendigits.tra');
  tes = load('pendigits.tes');

  % Iterate through all classes
  % After this loop, P will be a (nx10) matrix where
  % n = #test records and
  % P(i,c) = Probability of test-item i being in class c-1
  P = [];
  for i = 0:9
    %get all training-data for this class
    currentTrainingData = filterByClass(tra, i);

    % Compute mean and covariance matrix for this class
    [mu sigma] = meanAndCov(currentTrainingData);

    % Compute the probability density for all test-items
    % Add those as a column to P
    P = [P mvnpdf(tes(:,1:end-1), mu, sigma)];
  end
  
  % For each row get the index that has the highest value (probability)
  [maxValue maxIndex] = max(P,[],2);

  % Index begins with 1, our classes with 0
  maxIndex = maxIndex - 1;

  dlmwrite('recognition_results.mat', [maxIndex tes(:,end)], ' ');

end

% Input: (nx17)-Matrix with labled data (label in last column)
% Output: all rows having class c (without label column)
function r = filterByClass(data, c)
  r = data(ismember(data(:,end),c),1:end-1);
end

% Computes mean and covariance based on matrix with observations
% Input: rows with training data for one class
% Output: Mean (row vector) and covariance matrix (16x16)
function [m c] = meanAndCov(data)
  m = mean(data);
  c = cov(data);
  c = c + (0.0001 * eye(size(c)));
end