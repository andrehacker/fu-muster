%pkg load geometry

% Nearest Neighbor Function
function nn(trainingLimit, testingLimit, useknn, k, startfeat, endfeat)

  % Configuration variables
  trainingFile = 'pendigits-training.txt';
  testingFile  = 'pendigits-testing.txt';
  trainingFeatureFile = 'database.mat';
  trainingPredictedFile = 'database_nums.mat';
  testingFeatureFile = 'testdata.mat';
  testingPredictedFile = 'testdata_nums.mat';
  fileRecognitionResults = 'recognition_results.mat';
  enableFeatureCaching = 0;  % 1 = Cache Files, 0 = Disable caching

  fprintf('\n** TESTRUN **\n\n');
  fprintf('training-rows: %d\ntest-rows: %d\n', trainingLimit, testingLimit);
  fprintf('use k-nn? %d\n', useknn);
  fprintf('k: %d\n', k);
  fprintf('\n');

  % --------------
  % 1. Feature Extraction
  % --------------
  %
  % Calculate features for training data
  tic()
  if enableFeatureCaching == 0 || isempty(file_in_loadpath(trainingFeatureFile)) || isempty(file_in_loadpath(trainingPredictedFile))
    train = dlmread(trainingFile);
    train = train(1:trainingLimit,:,:);  % take only first trainingLimit rows
    database = calculateFeatures(train, startfeat, endfeat);
    database_nums = train(:,17);
    save (trainingFeatureFile, 'database');
    save (trainingPredictedFile, 'database_nums');
  else
    fprintf('use cached training data\n');
    load (trainingFeatureFile, 'database');
    load (trainingPredictedFile, 'database_nums');
  end;
  mytoc('Seconds Training Feature Extraction: ');

  % Calculate features for test data
  tic()
  if enableFeatureCaching == 0 || isempty(file_in_loadpath(testingFeatureFile)) || isempty(file_in_loadpath(testingPredictedFile))
    test = dlmread(testingFile);
    test = test(1:testingLimit,:,:);  % take only first testingLimit rows
    testdata = calculateFeatures(test, startfeat, endfeat);
    testdata_nums = test(:,17);
    save (testingFeatureFile, 'testdata');
    save (testingPredictedFile, 'testdata_nums');
  else
    fprintf('use cached test data\n');
    load (testingFeatureFile, 'testdata');
    load (testingPredictedFile, 'testdata_nums');
  end;
  mytoc('Seconds Test Feature Extraction: ');


  % --------------
  % 2. PREDICTION
  % --------------
  %
  % Implements linear NN-Search
  % O(n*m) runtime with n objects to predict and m objects in trainingspace

  tic()
  isVsShould = zeros(size(testdata, 1), 2);

  parfor i = 1:size(testdata, 1)
    if useknn == 0
      isVsShould(i,:) = [linearNNSearch(database, database_nums, testdata(i,:)), testdata_nums(i)];
    else
      isVsShould(i,:) = [linearKNNSearch(k, database, database_nums, testdata(i,:)), testdata_nums(i)];
    end;
  end;

  % Store results to file. Can be analyzed using analyze.m
  save (fileRecognitionResults, 'isVsShould');
  mytoc('Seconds Prediction: ');
  
end


% --------------
% FUNCTIONS
% --------------

% Implements Linear Nearest Neighbor Search for 1 new item
% n test items, f features
% database = (n x f) matrix. Feature space of test data
% database_nums = (n x 1) matrix. Classes of test data
% new_pattern = (1 x f) feature vector of obect to be classified
% return value: predicted class
function result = linearNNSearch(database, database_nums, new_pattern)
  min = inf;
  index = 1;
  for i = 1:size(database, 1)
    d = sqDistance(database(i,:)', new_pattern');
    %d = distancePoints(database(i,:), new_pattern);
    if d < min
      min = d;
      index = i;
    end;
  end;
  result = database_nums(index);
end

% Same as linearNNSearch, but k-Nearest Neighbor search
% Uses the simple approach to sort the result and take first k
% TODO: Add weighting (closed neighbors are weighted more).
% E.g. give each neighbor the weigth 1/d (according to Wikipedia)
function result = linearKNNSearch(k, database, database_nums, new_pattern)
  n = size(database, 1);

  % calculate distances to all test-objects.
  % Store distance (left) and id (to get class after sorting) (right)
  dist = [zeros(1,n); 1:n]';
  for i = 1:n
    dist(i, 1) = sqDistance(database(i,:)', new_pattern');
    %dist(i, 1) = distancePoints(database(i,:), new_pattern);
  end;
  firstk = sortrows(dist, 1); %sort by first column, take first k rows
  firstk = firstk(1:k,:);

  % count the frequency of the classes for the k nearest neighbors
  classFrequency = zeros(10,1); % what I really want is 0 to 9;-(
  for i = 1:k
    classFrequency(database_nums(firstk(i,2))+1) = classFrequency(database_nums(firstk(i,2))+1) + 1;  % attention, offset +1!
  end;

  % take the most frequent one. Attention: We have an offset (starting with class 1 instead of 0)
  [~, index] = max(classFrequency);
  result = index - 1;    % minimum returns index as 2nd argument

end

% Method extracting the features for one object
% Parameter data: Ursprungsdatei eingelesen in n x 17 Matrix
function result = calculateFeatures(data, startfeat, endfeat)
  pens = zeros(size(data, 1), 8, 2);  % (x,y) in einer intuitiveren darstellung
  vecs = zeros(size(data, 1), 7, 2);  % Die 7 freien Vektoren zwischen den 8 Punkten
  feat = zeros(size(data, 1), 19);     % Merkmale


  for k = 1:size(data, 1)
    for n = 1:2:16
      pens(k, ceil(n / 2), 1) = data(k, n);
      pens(k, ceil(n / 2), 2) = data(k, n + 1);
    end;
  end;


  % Freie Vektoren
  for k = 1:size(data, 1)
    for n = 1:7
      vecs(k,n,1) = vecs(k,n,1) + pens(k, n+1, 1) - pens(k, n, 1);
      vecs(k,n,2) = vecs(k,n,2) + pens(k, n+1, 2) - pens(k, n, 2);
    end;
  end;


  % Entfernung Start-Ende
  for k = 1:size(data, 1)
    a = pens(k,1,1:2);
    a = a(:)';
    b = pens(k,8,1:2);
    b = b(:)';
    feat(k,1) = norm(a - b);
  end;


  % Laenge des Schriftzugs
  for k = 1:size(data, 1)
    for n = 1:7
      a = pens(k,n,1:2);
      a = a(:)';
      b = pens(k,n+1,1:2);
      b = b(:)';
      feat(k,2) = feat(k,2) + norm(a - b);
    end;
  end;


  % (Start und Ende)/(Laenge)
  for k = 1:size(feat, 1)
    feat(k,3) = feat(k,1) / feat(k,2);
  end;


  % Winkel
  %for k = 1:size(data, 1)
  %  for n = 1:6
  %    a = vecs(k,n,1:2);
  %    a = a(:)';
  %    b = vecs(k,n+1,1:2);
  %    b = b(:)';
  %    feat(k,4) = feat(k,4) + vectors2angle(a, b);
  %  end;
  %end;


  result = [feat data(:,1:16)];

end




% --------------
% HELPERS
% --------------

function mytoc(timerDescription)
  %disp( ceil(toc()) )
  elapsed = toc();
  fprintf( '%s %.6f\n', timerDescription, elapsed );
  flush()
  %fprintf([timerDescription mat2str(ceil(toc()))]);
end

function flush
    drawnow('update');
end


% Winkel zwischen 2 freien Vektoren
%function result = vectors2angle(v1, v2)
%  result = rad2deg(normalizeAngle(vectorAngle(v1, v2), 0));
%end

function D = sqDistance(X, Y)
  D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
end


