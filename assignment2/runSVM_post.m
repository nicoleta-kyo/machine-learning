function [model, labels, kfold_loss, confusion_matrices] = runSVM_post(traindata, outputdata, testdata, weights)

    disp('Fitting SVM...')
    model = fitcsvm(traindata, outputdata, 'Weights', weights);
    %[labels,~] = predict(model, testdata);
    disp('Cross-validation...')
    cross_model = crossval(model);     %do k-fold cross validation    
    
    %fit posterior to get scores
    disp('Fitting Posterior...')
    score_model = fitSVMPosterior(model);
    [~, score] = predict(score_model, testdata);

    %calculate train priors
    count_pos=0;
    for datap=1:size(outputdata,1)
        if outputdata(datap,1) == 1
            count_pos = count_pos+1;
        end
    end
    prior_pos = count_pos/size(outputdata,1);
    prior_neg = 1 - prior_pos;

    %calculate class priors for snowy and not snowy
    prior_scores = zeros(size(testdata,1), 2);
    for datap=1:size(testdata,1)
        prior_scores(datap,1) = score(datap,1)/prior_neg*0.5714;  %multiply by test proportion for class 0
        prior_scores(datap,2) = score(datap,2)/prior_pos*0.4286;  %multiply by test proportion for class 1
    end

    %create new labels by choosing highest score
    labels = zeros(size(score,1), 1);
    for s=1:size(score,1)
        if(prior_scores(s,1) > prior_scores(s,2))
            labels(s) = 0;
        else
            labels(s) = 1;
        end
    end
    
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