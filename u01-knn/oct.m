pkg load geometry
%
% This is a script file, meaning, it does not start with a function
%

% --------------
% HELPERS
% --------------

% Helper function for Timer.
function mytoc(timerDescription)
  global total;
  %disp( ceil(toc()) )
  elapsed = toc();
  printf( '%s %.6f\n', timerDescription, elapsed );
  fflush(stdout);
  %printf([timerDescription mat2str(ceil(toc()))]);
endfunction

function flush
  fflush(stdout);
endfunction


% Winkel zwischen 2 freien Vektoren
function result = vectors2angle(v1, v2)
  result = rad2deg(normalizeAngle(vectorAngle(v1, v2), 0));
endfunction;


% --------------
% FUNCTIONS
% --------------

function D = sqDistance(X, Y)
  D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
endfunction;


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
    endif;
  end;
  result = database_nums(index);
endfunction;

% Same as linearNNSearch, but k-Nearest Neighbor search
% Uses the simple approach to sort the result and take first k
% TODO: Add weighting (closed neighbors are weighted more).
% E.g. give each neighbor the weigth 1/d (according to Wikipedia)
function result = linearKNNSearch(k, database, database_nums, new_pattern)
  n = size(database, 1);

  % calculate distances to all test-objects.
  % Store distance (left) and id (to get class after sorting) (right)
  dist = [zeros(1,n); 1:n]';
  min = inf;
  for i = 1:n
    dist(i, 1) = sqDistance(database(i,:)', new_pattern');
    %dist(i, 1) = distancePoints(database(i,:), new_pattern);
  end;
  firstk = sortrows(dist, 1)(1:k,:); %sort by first column, take first k rows

  % count the frequency of the classes for the k nearest neighbors
  classFrequency = zeros(10,1); % what I really want is 0 to 9;-(
  for i = 1:k
    classFrequency(database_nums(firstk(i,2))+1)++;  % attention, offset +1!
  end;

  % take the most frequent one. Attention: We have an offset (starting with class 1 instead of 0)
  [ignore, index] = max(classFrequency);
  result = index - 1;    % minimum returns index as 2nd argument

endfunction;

% Parameter data: Ursprungsdatei eingelesen in n x 17 Matrix
function result = calculateFeatures(data)
  pens = zeros(size(data, 1), 8, 2);  % (x,y) in einer intuitiveren darstellung
  vecs = zeros(size(data, 1), 7, 2);  % Die 7 freien Vektoren zwischen den 8 Punkten
  nums = data(:, 17);              % Das, was der Proband tatsaechlich schreiben wollte
  feat = zeros(size(data, 1), 20);     % Merkmale


  for k = 1:size(data, 1)
    for n = 1:2:16
      pens(k, ceil(n / 2), 1) = data(k, n);
      pens(k, ceil(n / 2), 2) = data(k, n + 1);
    end;
  end;


  % Freie Vektoren
  for k = 1:size(data, 1)
    for n = 1:7
      vecs(k,n,1) += pens(k, n+1, 1) - pens(k, n, 1);
      vecs(k,n,2) += pens(k, n+1, 2) - pens(k, n, 2);
    end;
  end;


  % Entfernung Start-Ende
  for k = 1:size(data, 1)
    a = pens(k,1,1:2)(:)';
    b = pens(k,8,1:2)(:)';
    feat(k,1) = distancePoints(a, b);
  end;


  % Laenge des Schriftzugs
  for k = 1:size(data, 1)
    for n = 1:7
      a = pens(k,n,1:2)(:)';
      b = pens(k,n+1,1:2)(:)';
      feat(k,2) += distancePoints(a, b);
    end;
  end;


  % (Start und Ende)/(Laenge)
  for k = 1:size(feat, 1)
    feat(k,3) = feat(k,1) / feat(k,2);
  end;


  % Winkel
  for k = 1:size(data, 1)
    for n = 1:6
      a = vecs(k,n,1:2)(:)';
      b = vecs(k,n+1,1:2)(:)';
      feat(k,4) += vectors2angle(a, b);
    end;
  end;

  result = [feat data(:,1:16)];


endfunction;


% Nearest Neighbor Function
function nn

  % Configuration variables
  trainingFile = 'pendigits-training.txt';
  testingFile  = 'pendigits-testing.txt';
  trainingLimit = 7494;  % max 7494
  testingLimit = 3498;    % max 3498
  k = 10;
  useknn = 0; % 0 = 1-NN, 1 = K-NN

  trainingFeatureFile = 'database.mat';
  trainingPredictedFile = 'database_nums.mat';

  testingFeatureFile = 'testdata.mat';
  testingPredictedFile = 'testdata_nums.mat';

  fileRecognitionResults = 'recognition_results.mat';

  enableFeatureCaching = 0;  % 1 = Cache Files, 0 = Disable caching

  printf('\n** TESTRUN **\n\n');
  printf('training-rows: %d\ntest-rows: %d\n', trainingLimit, testingLimit);
  printf('use k-nn? %d\n', useknn);
  printf('k: %d\n', k);

  printf('\n');

  % --------------
  % 1. Feature Extraction
  % --------------
  %
  % Calculate features for training data
  % store results in file (caching)
  tic()
  if enableFeatureCaching == 0 || isempty(file_in_loadpath(trainingFeatureFile)) || isempty(file_in_loadpath(trainingPredictedFile))
    train = dlmread(trainingFile);
    train = train(1:trainingLimit,:,:);  % take only first trainingLimit rows
    database = calculateFeatures(train);
    database_nums = train(:,17);
    save (trainingFeatureFile, 'database');
    save (trainingPredictedFile, 'database_nums');
  else
    printf('use cached training data\n');
    load (trainingFeatureFile, 'database');
    load (trainingPredictedFile, 'database_nums');
  end;
  mytoc('Seconds Training Feature Extraction: ');
  %['Took' char(toc()) 'seconds']

  % Calculate features for test data
  tic()
  if enableFeatureCaching == 0 || isempty(file_in_loadpath(testingFeatureFile)) || isempty(file_in_loadpath(testingPredictedFile))
    test = dlmread(testingFile);
    test = test(1:testingLimit,:,:);  % take only first testingLimit rows
    testdata = calculateFeatures(test);
    testdata_nums = test(:,17);
    save (testingFeatureFile, 'testdata');
    save (testingPredictedFile, 'testdata_nums');
  else
    printf('use cached test data\n');
    load (testingFeatureFile, 'testdata');
    load (testingPredictedFile, 'testdata_nums');
  end;
  mytoc('Seconds Test Feature Extraction: ');


  % --------------
  % 2. PREDICTION using k-NN
  % --------------
  %
  % Implements linear NN-Search
  % O(n*m) runtime with n objects to predict and m objects in testspace

  tic()
  isVsShould = zeros(size(testdata, 1), 2);

  for i = 1:size(testdata, 1)
    if useknn == 0
      isVsShould(i,:) = [linearNNSearch(database, database_nums, testdata(i,:)), testdata_nums(i)];
    else
      isVsShould(i,:) = [linearKNNSearch(k, database, database_nums, testdata(i,:)), testdata_nums(i)];
    endif;
  end;
  save (fileRecognitionResults, 'isVsShould');
  mytoc('Seconds Prediction: ');
endfunction

nn()

run analyze.m