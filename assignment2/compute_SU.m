function SU = compute_SU(feat1, feat2)

    %calculate joint entropy
    joint_feats = [feat1; feat2];
   
    joint_entropy = entropy(joint_feats);

    MI =entropy(feat1) + entropy(feat2) - joint_entropy; 
    SU = 2*MI/(entropy(feat1)+entropy(feat2));


end