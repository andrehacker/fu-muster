function solution
  % -------
  % TASK 1
  % -------
  %fid = fopen('task1-results.txt','w');
  %fprintf(fid, 'Raw Data Mean Success Rate: %.3f\n', fisher);
  %for i = 3:16
  %  fprintf(fid, 'PCA Dimensions: %.3f Mean Success Rate: %.3f\n', i, fisher_pca(i));
  %end
  %fclose(fid);

  % -------
  % TASK 2 Perceptron
  % -------
  items = 1000;
  fid = fopen('task2-results.txt','w');

  % a) Create test data
  [X, y, w] = linSepData(items, 4);
  
  % b) Learn using Perceptron
  [w2, steps, E] = perceptron(X, y, 10000);
  plotPerceptronError(E, 'task2-perceptron');

  % Compare original and learned weight
  fprintf(fid, 'Generated %d items\n', items );
  fprintf(fid, 'Required iterations: %d\n', steps);
  fprintf(fid, 'Original weigth: %s\n', mat2str(w,3) );
  fprintf(fid, 'Learned weigth: %s\n', mat2str(w2,3) );
  fclose(fid);

  % -------
  % TASK 3
  % -------
  % a) Create correctly classified data
  [X, y, w] = task3ImageData();

  % b) Learn weights
  [w2, steps, E] = perceptron(X, y, 10000);
  plotPerceptronError(E, 'task3-perceptron');
  fid = fopen('task3a-results.txt','w');
  fprintf(fid, 'Required iterations: %d\n', steps);
  fprintf(fid, 'Original weigth: %s\n', mat2str(w,3) );
  fprintf(fid, 'Learned weigth: %s\n', mat2str(w2,3) );
  fclose(fid);

  detectContour(w2);

end

% Learn the weights using PLA
% w = learned weight
% steps = number of needed steps
% E = Error for each iteration
function [w steps E] = perceptron(X, y, maxSteps)
  w = rand(1, size(X,2))*2-1;
  E = [[1:maxSteps]', zeros(maxSteps,1)]; % Track error for each iteration
  for i=1:maxSteps
    % Already finished?
    wrongIds = getWrongClassified(X, y, w);
    E(i,2) = size(wrongIds,1);
    if size(wrongIds, 1) == 0
      break
    end

    % Weight improvement (for any wrong classified)
    id = wrongIds(randi(size(wrongIds,1)),:);
    if y(id) == 1 % false negative (should be 1)
      w = w + X(id, :);
    else  % false positive
      w = w - X(id, :);
    end
  end
  steps = i;
  E = E(1:i,:);
end

% Predict using the weight w with threshold 0
function y = perceptronPredict(X, w)
  y = X*w';
  y(y>=0)=1;
  y(y<0)=0;
end

% Classify and test whether we all is classified correctly
% if not, we return the indicies of all wrong items
function wrongIds = getWrongClassified(X, y, w)
  % Predict
  p = perceptronPredict(X,w);
  diff = y - p;

  % Get wrong
  % any selects the rows different from 0
  wrongIds= [1:size(X,1)]';
  wrongIds = wrongIds(any(diff,2));
end

% Create linear separable labled data with bias
% X = random items
% y = label, based on w
% w = random weigth
% dim = dimension
function [X, y, w] = linSepData(rows, dim)
  X = [ones(rows,1) rand(rows, dim-1)*2-1];
  w = rand(1, dim)*2-1;
  y = perceptronPredict(X,w);
end

% Generate data for task3
function [X, y, w] = task3ImageData()
  X = dec2bin(0:2^9-1)-'0';
  X = [ones(size(X,1),1) X];
  % 0=black, 1=white, mid must be black for edge
  % This is a working weighting:
  w = [-1 1 1 1 1 -10 1 1 1 1];
  y = perceptronPredict(X,w);
end

% Detect contours in a sample image
function detectContour(w)
  % Circle (from Barbara Haupt)
  [xs,ys]=meshgrid(-100:100);
  I=zeros(size(xs));
  I(sqrt(xs.^2+ys.^2)<(0.3*size(xs,1)))=1;
  h = figure;
  imagesc(I); colormap('gray'); axis equal off;
  print(h,'-deps',['task3b-original.eps']);

  % Detect contours
  h = figure;
  X = zeros(size(xs));
  for i=2:size(xs,1)-1
    for j=2:size(xs,1)-1
      x = [1, ...
        I(i-1,j-1), ...
        I(i-1,j), ...
        I(i-1,j+1), ...
        I(i,j-1), ...
        I(i,j), ...
        I(i,j+1), ...
        I(i+1,j-1), ...
        I(i+1,j), ...
        I(i+1,j+1)];
      X(i,j) = perceptronPredict(x, w);
    end
  end
  imagesc(X); colormap('gray'); axis equal off;
  print(h,'-deps',['task3b-contours.eps']);
end

% Plot Perceptron Error rate
function plotPerceptronError(E, name)
  h = figure();
  hold on;
  xlabel('Iteration', 'FontSize', 15);
  ylabel('Wrong classified', 'FontSize', 15);
  plot(E(:,1),E(:,2));
  print(h,'-deps',[name '.eps']);
end