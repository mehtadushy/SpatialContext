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
%%%%%%%%%%%%%%%%
% 2D Histograms of keypoint appearance captured with SIFT and keypoint
% context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [  40;
                       80;
                       ];
contextDictionarySize = [10,20,40];
keypointDictionarySize = [50, 100, 200, 400];

accuracy_results = cell(length(contextDictionarySize)+1, length(neighbourhoodSize)*length(keypointDictionarySize)+1);

for k = 1: length(contextDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Context', contextDictionarySize(k));
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, (j-1)*length(neighbourhoodSize)+i+1} = sprintf('%dN %d_KD', neighbourhoodSize(i), keypointDictionarySize(j));
          end      
     end
end

%%
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(contextDictionarySize);

for k = 1: length(contextDictionarySize)
     for j = 1: length(keypointDictionarySize)
         acc_log = cell(1, n_sz);
         parfor i = 1:length(neighbourhoodSize)
            %fprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k));
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_directional_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d_directional_relative_context_neigh%d', contextDictionarySize(k), neighbourhoodSize(i))));
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
            acc_log{i} = accuracy_P(1);
         end    
         accuracy_results(k+1 ,(j-1)*n_sz+2:j*n_sz+1) = acc_log;

     end
end


outfname = fullfile( data_dir, 'results', 'Directional2DContextHist_accuracy.mat');
save(outfname, 'accuracy_results');












