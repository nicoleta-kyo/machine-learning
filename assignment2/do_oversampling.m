function [oversampled_train, oversampled_output] = do_oversampling(train, output)
    
%find data points with label 0 and add them to an array
rep_negs = zeros(1920, size(train,2));
count_neg = 0;
for datap=1:size(train,1)
    if (output(datap,1) == 0)
        count_neg=count_neg+1;
        rep_negs(count_neg,:) = train(datap,:);
    end
end

%create new train set with more 0s
oversampled_train = [train; rep_negs];
%updata output labels for train set
oversampled_output = [output; zeros(count_neg,1)];

end
