function weights = update_weights_using_annotation_confidence(traindata, traindata_nan)
%%%expand dataset biasing the data points based on label confidence

confidence = csvread('annotation_confidence.csv',1,1);


%count NaNs and change annotation confidence
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

for datap=1:length(number_nans) 
    if (number_nans(datap) > mean(number_nans(457:end)) )
        confidence(datap) = 0.33;
    end
end


%update weights by using confidence
weights= zeros(size(traindata,1),1);
for datap=1:size(traindata,1)
    if confidence(datap,1) == 1 
         weights(datap) = 1; 
    end
    if confidence(datap,1) == 0.66
        weights(datap) = 0.66;        
    end
    if confidence(datap,1) == 0.33
        weights(datap) = 0.33;
    end
end



end





        