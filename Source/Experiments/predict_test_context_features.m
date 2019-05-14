%Using LibSVM Train
%This file too is to be run cell wise
image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');
%%

%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
% Cluster the features, assign labels and create histograms of the features
weights = [ %SIFT   %Spatial Context  %Orientation Context
              1          5               0;
              1          0               5;
              1          5               5;
              1          4               8;
              1          1               0;
              1          0               1;
              1          1               1;];
neighbourhoodSize =   [%20;
                       %25;
                       %30;
                       55;
                       80;
                       ];
dictionarySize = [800, 2000, 8000, 15000];  %[50, 100, 200, 400];

accuracy_results = cell(length(dictionarySize)+1, length(neighbourhoodSize)*size(weights,1)+1);


for k = 1:length(dictionarySize)
   params.dictionarySize = dictionarySize(k);
   accuracy_results{ k+1, 1} = sprintf('%d Dic',dictionarySize(k));
   for i = 1:length(neighbourhoodSize)
    for w = 1:size(weights,1)
        accuracy_results{1, (i-1)*(size(weights,1))+1+w} = sprintf('%dN %1.1f|%1.1f|%1.1f ',neighbourhoodSize(i), weights(w,1), weights(w,2), weights(w,3));
    end
   end
end




%%
for k = 1:length(dictionarySize)
   for i = 1:length(neighbourhoodSize)
       acc_log = cell(1, size(weights,1));
       fprintf('Predicting With %d Dictionary %d Neigh Size', dictionarySize(k), neighbourhoodSize(i));
       for w = 1:size(weights,1)
            sift_weight = weights(w,1);
            spatial_weight = weights(w,2);
            orientation_weight = weights(w,3);
            if (sift_weight > 0)
                sift_string = sprintf('_siftWght_%1.1f',sift_weight);
            else
                sift_string = '';
            end
            if (spatial_weight > 0)
                spatial_string = sprintf('_spatialWht_%1.1f',spatial_weight);
            else
                spatial_string = '';
            end
            if (orientation_weight > 0)
                orientation_string = sprintf('_orientationWht_%1.1f',orientation_weight);
            else
                orientation_string = '';
            end

            inFName = fullfile(data_dir, sprintf('soft_histograms_%d%s%s%s%s', dictionarySize(k), sift_string, spatial_string, orientation_string, sprintf('_keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i))) );
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
            test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{w} = accuracy_P(1);
            
%             confusion_matrix = cell(length(dirlist)+1,length(dirlist)+1);
%             for i = 1:length(dirlist)
%                 confusion_matrix{i+1,1} = dirlist{i};
%                 num_i_labels = sum(test.labels == i);
%                 for g = 1:length(dirlist)
%                     confusion_matrix{1,g+1} = dirlist{g};
%                     num_j_labels = sum(predict_label_P(test.labels == i) == g);
%                     confusion_matrix{i+1,g+1} = num_j_labels/num_i_labels;
%                 end
%             end
            
            %outfname = fullfile( data_dir, 'results',  sprintf('confusion_matrix_%s_with%s.mat', sprintf('_%d_KeyptDic',keypointDictionarySize(j)), sprintf('_%d_DenseDic_%d_%d_NeighSize_NeighHistDic%d', dictionarySize(dic), sigmaFactors(sf,1), sigmaFactors(sf,2), neighborhoodHistogramDictionarySize(k))));
            %save(outfname, 'confusion_matrix');
            
        end
        accuracy_results(k+1 ,(i-1)*(size(weights,1))+2:(i)*(size(weights,1))+1) = acc_log;
    end
end


outfname = fullfile( data_dir, 'results', 'large_dictionary_context_sift_accuracy.mat');
save(outfname, 'accuracy_results');







