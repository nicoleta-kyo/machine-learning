clear all
addpath 'C:\nikyk\university of sussex\Year 2\Machine Learning\ass1';
indata = csvread('data167184.csv');

%normalise input data
indata_norm = (indata(:,1:5)-mean(indata(:,1:5)))./std(indata(:,1:5));
indata_norm = [indata_norm indata(:,6)];
%do pca
C = cov(indata_norm(:,1:5));
[U,lambda] = eig(C);
data_reduced = indata_norm(:,1:5)*U(:,3:5);
%add target data to data
data_reduced = [data_reduced indata_norm(:,6)];

data = data_reduced;
%add biases to the features


%set up k-fold cross validation
cv = cvpartition(size(indata,1), 'KFold', 5);
err = zeros(cv.NumTestSets,2);
nets = cell(3,3);

%set training function
trainFcn = 'trainbr';
%set size and number of hidden layers
hiddenLayerSize = [15 15];
%number of layers
numl = [2 3];
%size of layer
sizel = [5 10 15 20];
cvErr = zeros(length(sizel),length(numl));
numLcvErr = zeros(2, length(sizel));

%loop through number of layers
for nl=1:length(numl)
    numLayers = numl(nl);
%create hiddenLayerSize vector by looping through sizes of layers
for j=1:length(sizel)
    hiddenLayerSize = repelem(sizel(j), numLayers);
%perform cross validation for the chosen topology
for i = 1:cv.NumTestSets
    %get indices of data for cross validation
    trIdx = cv.training(i);
    teIdx = cv.test(i);
    %split data by indices
    train = indata(trIdx, :);
    test = indata(teIdx, :);
    data = [train; test];
    input = data(:,1:5)';
    target = data(:,6)';
    [nets{i,1},err(i,1)] = runNN(input, target, trainFcn, hiddenLayerSize);
end
%show what's the performance over the folds for each size of layer
cvErr(j,nl) = mean(err);
end
%show what's the mean for each number of layers
numLcvErr(1,nl) = numLayers;
numLcvErr(2,nl) = mean(cvErr(:, nl));
end



