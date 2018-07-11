function [train_reduced, test_reduced] = feature_selection_SU(train, output, test)
%take features that contribute to the classes the most
%and are uncommon for the two classes


feats_tokeep1 = do_feature_selection_for_class(train, output, 1);
feats_tokeep2 = do_feature_selection_for_class(train, output, 0);
feats_tokeep_all = setdiff(feats_tokeep1, feats_tokeep2);  
feats_common = intersect(feats_tokeep1, feats_tokeep2);
%get different plus 4/5 of common
feats_tokeep_all = [feats_tokeep_all; feats_common(1:floor(length(feats_common)/5)*4,:)];




%%%%%%%%%%create reduced data based on SU %%%%%

train_reduced = zeros(size(train,1), length(feats_tokeep_all));
test_reduced = zeros(size(test,1), length(feats_tokeep_all));
feat_index = 0;
for feat=1:length(feats_tokeep_all)
    feat_index=feat_index+1;
    feature = feats_tokeep_all(feat,1);  %get feature index
    train_reduced(:,feat_index) = train(:,feature);  %add feature to reduced train set
    test_reduced(:,feat_index) = test(:,feature);     %add feature to reduced test set
    
end

end