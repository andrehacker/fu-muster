function new_fisher
  % -------
  % TASK 1
  % -------
  % Load the data
  tra = load('pendigits.tra');
  tes = load('pendigits.tes');
  fid = fopen('task1-results.txt','w');
  
  % We standardize the data to adjust for large differences in absolute
  % feature values
  tmp = standardize(tra(:,1:end-1));
  tra = [tmp tra(:,end)];
  tmp = standardize(tes(:,1:end-1));
  tes = [tmp tes(:,end)];

  % here we learn the fisher discriminant using the training data and test
  % it with the test data. We get a matrix of pairwise recognition success
  % rates for the pendigits
  m = test_fisher(tra, tes);
  
  % calculate the avarage success rate
  sr = fisher_success_rate(m);
  fprintf(fid, 'Raw Data Mean Success Rate: %.3f%%\n', 100*sr);

  % Visualize some digits
  pc = principalComponents(tra(:,1:end-1));
  tmp = transformData(pc, tra(:,1:end-1), 16);
  for i=1:10
    h = figure('NumberTitle','on');
    hold on;
    plot( tmp(i,1:2:end-1), tmp(i,2:2:end) );
    title(['Training item ' mat2str(i) ' Digit ' mat2str(tra(i,end))], 'FontSize', 30);
    print(h,'-deps',['task1-transformed-digit-' mat2str(i) '.eps']);
  end
  
  
  for dim = 1:16
    save_tra = tra;
    save_tes = tes;
    
    pcs = principalComponents(tra(:,1:end-1));
    tmp = transformData(pcs, tra(:,1:end-1), dim);
    tra = [tmp tra(:,end)];
    
    tmp = transformData(pcs, tes(:,1:end-1), dim);
    tes = [tmp tes(:,end)];
    
    if dim == 10
      m = test_fisher(tra, tes)
    else
      m = test_fisher(tra, tes);
    end
    sr = fisher_success_rate(m);
    fprintf(fid, 'PCA Dimensions: %.0f Mean Success Rate: %.3f%%\n', dim, 100*sr);
    tra = save_tra;
    tes = save_tes;
  end
end


function r = fisher_success_rate(s)
  r = 0;
  for n = 1:9
    for m = n+1:10
      r = r + s(n,m);
    end
  end
  r = r / (9*10/2);
end

% compute and test the fisher discriminant
function result = test_fisher(tra, tes)

  result = zeros(10,10);

  for n = 1:9
    for m = n+1:10
      i = n - 1;
      j = m - 1;
      classes = [i j];
      sampel0 = filterByClass(tra, i);
      sampel0 = sampel0(:,1:end-1);
      sampel1 = filterByClass(tra, j);
      sampel1 = sampel1(:,1:end-1);

      testdata = [filterByClass(tes, i); filterByClass(tes, j)];

      % compute multivariate distribution parameters in feature space
      mu0 = mean(sampel0);
      sigma0 = cov(sampel0);
      mu1 = mean(sampel1);
      sigma1 = cov(sampel1);

      a = fisherDiscriminant(mu0, sigma0, mu1, sigma1);

      % compute new mean/variance for 1d distribution on the fisher 
      % discriminant
      psigma0 = a * sigma0 * a';
      psigma1 = a * sigma1 * a';
      pmu0 = project(mu0, a);
      pmu1 = project(mu1, a);
    
      % test with test data
      projection = project(testdata(:,1:end-1), a);

      % compute the propabilties for each test sample and the two classes.
      P = [gaussDensity(pmu0, psigma0, projection) ...
           gaussDensity(pmu1, psigma1, projection) ];

      [maxValue maxIndex] = max(P,[],2);
      
      % contains the predicted classes
      class = classes(maxIndex)';

      % computes the success rate of prediction
      success = analyze([class testdata(:,end)]);
      result(n,m) = success;
    end
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


function success = analyze(isVsShould)

  hit = 0;
  miss = 0;

  misses = zeros(10,10);
  hits = zeros(1,10);

  for k = 1:size(isVsShould, 1)
    if isVsShould(k,1) == isVsShould(k,2)
      hit = hit +1;
      hits(isVsShould(k,2)+1) = hits(isVsShould(k,2)+1) + 1;
    else
      miss = miss + 1;
      misses(isVsShould(k,1)+1,isVsShould(k,2)+1) = misses(isVsShould(k,1)+1,isVsShould(k,2)+1) + 1;

    end;
  end;

  success = 1-(miss/(hit+miss));
end


function p = gaussDensity(mu, sigma, data)
  normalize = 1 / (sqrt(2*pi) * sigma);
  p = zeros(size(data,1), 1);
  for i=1:size(data,1)
    p(i) = normalize * exp( -(data(i)-mu)^2 / (sigma^2) );
  end

end


function r = filterByClass(samples, c)
  r = samples(ismember(samples(:,end),c),:);
end


function direction = fisherDiscriminant( mean1, covar1, mean2, covar2 )
  u = (mean1 - mean2) / (covar1 + covar2);
  direction = u / norm(u);
end


function p = project(x, a)
  if (size(x,1)>1)
    p = dot(x, repmat(a, size(x,1),1), 2);
  else
    p = dot(x,a);
  end
end
