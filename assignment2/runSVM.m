function [model, labels, kfold_loss, confusion_matrices] = runSVM(traindata, outputdata, testdata)

    disp('Fitting SVM...')
    model = fitcsvm(traindata, outputdata, 'Box Constraint', );
    [labels,~] = predict(model, testdata);
    disp('Cross-validation...')
    cross_model = crossval(model);     %do k-fold cross validation    

    
    %CALCULATE TRAIN AND TEST ERROR RATE
    losses = zeros(10,2);
    confusion_matrices = cell(10,1);
    for fold=1:10
        disp(strcat('Calculating error for fold No.', num2str(fold)));
        trainIndx = training(cross_model.Partition, fold);
        testIndx = test(cross_model.Partition, fold);
        compact_model = cross_model.Trained{fold};             %get compact model
        losses(fold,1) = loss(compact_model, traindata(trainIndx,:), outputdata(trainIndx));
        losses(fold,2) = loss(compact_model, traindata(testIndx,:), outputdata(testIndx));
        %create confusion matrix
        [lab,~] = predict(compact_model, traindata(testIndx,:));
        conf_matrix = confusionmat(outputdata(testIndx,:), lab, 'Order', [1 0]);
        confusion_matrices{fold} = conf_matrix;
    end
    kfold_loss(1,1) = mean(losses(:,1));
    kfold_loss(2,1) = mean(losses(:,2));
   
    
    disp(strcat('Train error:', num2str(kfold_loss(1,1))));
    disp(strcat('Test error:', num2str(kfold_loss(2,1))));

    
    %CALCULATE PROPORTION OF POSITIVE LABELS
    count_pos = 0;
    for i=1:length(labels)
        if (labels(i,1) ~= 0)
            count_pos=count_pos+1;
        end
    end
    prop_pos = count_pos/size(labels,1)*100;
    disp(strcat('Positive labels: ', num2str(prop_pos)));
    
    


end