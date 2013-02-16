function fisher

  % Load input into (nx17) Matrix.
  % last column stores the class number
  tra = load('pendigits.tra');
  tes = load('pendigits.tes');

  % Iterate through classes
  result = zeros(10,10);
  for i=0:9
    for j=0:i-1
      samples0 = filterByClass(tra, i);
      samples1 = filterByClass(tra, j);
      test01 = filterByClass(tes, [i j]);
      result(i+1,j+1) = fisher2(samples0, samples1, test01, [i j]);
      result(j+1,i+1) = result(i+1,j+1);
    end
  end
  result = result * 100
  meanSuccessRate = mean(mean(result))

  dlmwrite('fisher_results.mat', result, ' ');

  % Simple examples that are plotted
  fisher2dTest();

end

% Test-function to generate some simple 2d classification problems
function fisher2dTest

  % Example
  samples1 = [0 0; 1 1; 2 2];
  samples2 = [2 4; 3 3; 4 2];
  fisher2(samples1, samples2, [0 0], [1 2], 'fisher1');

  % Example with s1 = s2
  samples1 = [0 0; 1 1; 2 2];
  samples2 = [1 -1; 2 0; 3 1];
  fisher2(samples1, samples2, [0 0], [1 2], 'fisher2');

  % Example with sigma1 having many zeros
  samples1 = [0 0; 1 1; 2 2; 0 2; 2 0];
  samples2 = [1 -1; 2 0; 3 1];
  fisher2(samples1, samples2, [0 0], [1 2], 'fisher3');

end


% Classification function for two classes
% Computes the fisher discriminant, returns success rate
function success = fisher2(samples1, samples2, testdata, classes, name)

  % Compute mu's and sigma's (covariance matrices)
  mu1 = getMean(samples1(:,1:end-1));
  sigma1 = getCovar(samples1(:,1:end-1), mu1);
  mu2 = getMean(samples2(:,1:end-1));
  sigma2 = getCovar(samples2(:,1:end-1), mu2);

  % Determine optimum projection using fisher criteria
  if sigma1 == sigma2
    % add noise
    sigma2 = 0.01 * rand(size(sigma2,1), size(sigma2,2));
  end
  a = (mu1 - mu2) * (sigma1 - sigma2)^(-1);
  % Normalize to lenght 1
  a = a / sqrt(a*a');

  % Compute mu's and sigma's for projection
  % In script we have column-vectors, here row-vectors
  psigma1 = a * sigma1 * a';
  psigma2 = a * sigma2 * a';
  pmu1 = project(mu1, a);
  pmu2 = project(mu2, a);

  % --------------
  % CLASSIFY
  % --------------
  success = fisherClassify([pmu1 pmu2], [psigma1 psigma2], a, testdata, classes);

  % Plot 2d example
  if size(a,2)==2
    plotfisher(samples1(:,1:end-1), samples2(:,1:end-1), ...
      mu1, sigma1, mu2, sigma2, ...
      pmu1, psigma1, pmu2, psigma2, a, name);
  end

end


% Classify based on two normal distributions
% uses projection on fisher discriminant
% Returns success rate
function success = fisherClassify(means, sigmas, disc, testdata, classes)
  % Project all to fisher discriminant
  projection = project(testdata(:,1:end-1), disc);

  % Get prob. of each class and projection
  P = [gaussDensity(means(1), sigmas(1), projection) ...
    gaussDensity(means(2), sigmas(2), projection) ];

  % Take class with max. prob.
  % For each row get the index that has the highest value (probability)
  [maxValue maxIndex] = max(P,[],2);

  % Replace index with number(class) it represents
  maxIndex = classes(maxIndex)';

  % Write result (predicted and real class)
  % dlmwrite('recognition_results.mat', [maxIndex testdata(:,end)], ' ');

  success = analyze([maxIndex testdata(:,end)]);

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


% Return projection from x on a (row vectors)
% x: rows (1 or many) with items to project
% a: row vector where to project on
function p = project(x, a)
  if (size(x,1)>1)
    p = dot(x, repmat(a, size(x,1),1), 2);
  %  p = p * a
  else
    p = dot(x,a);  % Just take the scaling factor
    %p = (dot(x,a)/dot(a,a)) * a;
  end

end

function p = gaussDensity(mu, sigma, data)
  normalize = 1 / (sqrt(2*pi) * sigma);
  p = zeros(size(data,1), 1);
  for i=1:size(data,1)
    p(i) = normalize * exp( -(data(i)-mu)^2 / (sigma^2) );
  end

end

% Input: (nx17)-Matrix with labled samples
% Output: rows with class c (without label column)
function r = filterByClass(samples, c)
  r = samples(ismember(samples(:,end),c),:);
end

% Computes mean based on matrix with observations
% Input: rows with training data for one class
% Output: Mean (row vector)
function m = getMean(data)
  m = 1/size(data,1) * sum(data);
end

% Compute Covariance matrix for 16-d samples
function s = getCovar(samples, mu)
  % Normalize and compute covar
  samples = samples - (ones(size(samples,1),1) * mu);
  s = samples' * samples;
  s = s / size(samples,1);

  % make some noise;)
  s = s + (0.0001 * eye(size(s)));
end


function plotfisher(samples1, samples2, m1, s1, m2, s2, ...
  pm1, ps1, pm2, ps2, a, name)
  h = figure;
  % http://www.mathworks.de/de/help/matlab/ref/colorspec.html
  % http://www.mathworks.de/de/help/matlab/ref/linespec.html
  scatter(samples1(:,1), samples1(:,2), 'bo', 'filled', 'Displayname', 'Class 1');
  hold on;
  scatter(samples2(:,1), samples2(:,2), 'g^', 'filled', 'Displayname', 'Class 2');
  scatter(m1(1), m1(2), 200, 'b*', 'Displayname', 'Mu Class 1');
  scatter(m2(1), m2(2), 200, 'g*', 'Displayname', 'Mu Class 2');
  scatter(pm1(1), pm1(2), 200, 'b+', 'Displayname', 'Mu Projection C1');
  scatter(pm2(1), pm2(2), 200, 'g+', 'Displayname', 'Mu Projection C2');
  quiver(0,0,a(1),a(2), 'Displayname', 'Fisher Disc.');

  legend('show');

  print(h,'-dpng',[name '.png']);
  %print(h,'-dpng',['fisher-' datestr(now, 'yyyy-mm-dd HH:MM:SS') '.png']);
end