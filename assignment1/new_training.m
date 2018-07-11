cv = cvpartition(size(data,1), 'KFold', 5);
err = zeros(cv.NumTestSets,2);
nets = cell(3,3);

%set training function
trainFcn = 'trainbr';
%set size and number of hidden layers
hiddenLayerSize = [35];

for i = 1:cv.NumTestSets
    %get indices of data for cross validation
    trIdx = cv.training(i);
    teIdx = cv.test(i);
    %split data by indices
    train = data(trIdx, :);
    test = data(teIdx, :);
    findata = [train; test];
    input = findata(:,1:3)';
    target = findata(:,4)';
    [nets{i,22},err(i,22)] = runNN(input, target, trainFcn, hiddenLayerSize);
end
err(6,22) = mean(err(1:5,22));