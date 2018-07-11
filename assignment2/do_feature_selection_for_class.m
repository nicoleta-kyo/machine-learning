function [feats_tokeep] = do_feature_selection_for_class(traindata_normed, outputdata, class)


    %%%%%%%feature selection for class class%%%%%%%%%
    train_outputdata = [traindata_normed outputdata];
    class_data_indx = train_outputdata(:,end) == class;     %get only data for class
    class_data = train_outputdata(class_data_indx,1:end-1);
    class_output = normc(class_data(:,end));                   
    SU = zeros(size(class_data, 2),2);
    for feat_index=1:size(class_data, 2)
        feature=class_data(:,feat_index);
        SU(feat_index,1) = compute_SU(feature, class_output);
        SU(feat_index,2) = feat_index;
    end
    feats_tokeep = cell(size(class_data,2),1);
    [~, indx]= sort(SU(:,1), 'descend');
    sorted_SU = SU(indx,:);


    feat1=1;             %compare everything only to first feature for now
    SU_joints=[];
    feat1_index = sorted_SU(feat1, 2);               %get index of feature1 in sorted_SU
    feature1 = class_data(:, feat1_index);   %get actual feature1
    for feat2=feat1+1:size(class_data,2)-1
        feat2_index = sorted_SU(feat2, 2);           %get index of feature2 in sorted_SU
        feature2 = class_data(:, feat2_index);  %get actual feature2
        SU_joint = zeros(1,2);
        SU_joint(1,1) = compute_SU(feature1, feature2);
        SU_joint(1,2) = feat2_index;
        SU_joints = [SU_joints; SU_joint];
    end
    %get all of the features
     feats_tokeep = SU_joints(:, 2);
        



end
    


        

