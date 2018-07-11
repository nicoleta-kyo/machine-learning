function [expanded_traindata, expanded_outputdata] = expand_data(traindata, outputdata, traindata_nan)
%%%expand dataset biasing the data points based on label confidence

confidence = csvread('annotation_confidence.csv',1,1);


%count NaNs and for data point in the additional training set
number_nans = zeros(size(traindata_nan,1),1);
for datap=1:size(traindata_nan,1)
    nans = 0;
    for f=1:size(traindata_nan,2)
        if (isnan(traindata_nan(datap, f)))
            nans=nans+1;
        end
    end
    number_nans(datap) = nans;
    
end

%if the number of NaNs is less than the mean, set confidence to 0.33
for datap=1:length(number_nans) 
    if (number_nans(datap) > mean(number_nans(457:end)) )
        confidence(datap) = 0.33;
    end
end


%create expanded dataset: copy data points twice if confidence is 1,
%otherwise copy only once
start_ind = 0;
expanded_traindata = zeros(1718+size(traindata,1),size(traindata,2));
expanded_outputdata = zeros(1718+size(outputdata,1),size(outputdata,2));
for datap=1:size(traindata,1)
    if confidence(datap,1) == 1 
       expanded_traindata(start_ind+1,:) = traindata(datap,:);
       expanded_traindata(start_ind+2,:) = traindata(datap,:);
       expanded_outputdata(start_ind+1,:) = outputdata(datap,:);
       expanded_outputdata(start_ind+2,:) = outputdata(datap,:);
       start_ind = start_ind+2;
    else
       expanded_traindata(start_ind+1,:) = traindata(datap,:);
       expanded_outputdata(start_ind+1,:) = outputdata(datap,:);
       start_ind = start_ind+1;
    end
end

end





        