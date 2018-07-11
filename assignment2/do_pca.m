function [traindata_pca, testdata_pca] = do_pca(traindata_normed, testdata_normed)

%     all_data = [traindata_normed; testdata_normed];
%     start_index = size(all_data,2) - pca_param +1;
%     C = cov(all_data);
%     [U,lambda] = eig(C);
%     all_data_pca = all_data*U(:,start_index:end);
%     traindata_pca = all_data_pca(1:size(traindata_normed,1),:);
%     testdata_pca = all_data_pca((size(traindata_normed,1)+1):end,:);


      [COEFF, SCORE] = pca(traindata_normed);
      traindata_pca = SCORE;
      testdata_pca = testdata_normed*COEFF;
      


end