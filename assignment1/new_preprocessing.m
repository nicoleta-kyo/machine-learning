clear all
indata = csvread('data167184.csv');

%find features with lowest variance and remove them
data = zeros(1000,3);
var1 = cell(1,2);
var1{1,1} = 1;
var1{1,2} = var(indata(:,1));
var2 = cell(1,2);
var2{1,1} = 2;
var2{1,2} = var(indata(:,2));
var3 = cell(1,2);
var3{1,1} = 3;
var3{1,2} = var(indata(:,3));
var4 = cell(1,2);
var4{1,1} = 4;
var4{1,2} = var(indata(:,4));
var5 = cell(1,2);
var5{1,1} = 5;
var5{1,2} = var(indata(:,5));
vars = {var1, var2, var3, var4, var5};
variances = [var1{1,2}, var2{1,2}, var3{1,2}, var4{1,2}, var5{1,2}];
variances = sort(variances);
for i=1:3
    for j=1:5
        if vars{i}{2} == variances(j)
            data(:,i) = indata(:, vars{i}{1});
        end
    end
end

%do pca on the data
[COEFF,SCORE,latent] = pca(indata_norm(:,1:3));
data = [SCORE, indata_norm(:,4)];

