
% --------------
% Classifier using multivariant normal distribution
% SUCCESS RATE: 0.959
% --------------
function covar2
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
    train = filterByClass(tra, i);

    % Compute mean and covariance matrix for this class
    mu = getMean(train);
    sigma = getCovarMatrix(train, mu);

    % Compute the probability density for all test-items
    % Add those as a column to P
    %P = [P mvnpdf(tes(:,1:end-1), mu, sigma)];
    P = [P multivariateDensity(tes(:,1:end-1), mu, sigma)];
  end
  %disp(P(1:1,:));
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

% Computes mean based on matrix with observations
% Input: rows with training data for one class
% Output: Mean (row vector)
function m = getMean(data)
  m = 1/size(data,1) * sum(data);
end

% Computes covariance matrix based on matrix with observations
% Input: rows with training data for one class
% Output: Covariance matrix (16x16)
function c = getCovarMatrix(data, mu)
  c = zeros(16,16);
  mu=mu';
  for i=1:size(data,1)
    xi = data(i,:)';
    c = c + (xi-mu) * (xi-mu)';
  end

  % Normalize
  c = c / size(data,1);

  %c = cov(data); % it would be so simple;)

  % make some noise;)
  c = c + (0.0001 * eye(size(c)));
end

% Multivariate normal distribution density function (pdf)
% produces same output as mvnpdf, based on the formula in the rojas tutorial
function p = multivariateDensity(data, mu, sigma)
  mu = mu'; % We work with column vectors here
  normalize = 1/sqrt(det(2 * pi * sigma));
  p = zeros(size(data,1), 1);
  for i=1:size(data,1)
    cur = data(i,:)';
    p(i) = normalize * exp( -0.5 * (cur-mu)' * inv(sigma) * (cur-mu) );
  end
end