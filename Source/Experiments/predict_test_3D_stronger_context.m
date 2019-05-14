%Using LibSVM to Train and test SVMs.
%This does not use an explicit kernel 
image_dir = '../Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign';
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
% 3D Histograms of keypoint appearance captured with SIFT and keypoint
% context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ 40;
                       80;
                       ];
coarseDictionarySize = 40;%[20, 40, 50]; %Coarse Dictionary Size
contextDictionarySize = [10 ,20, 40];
keypointDictionarySize = [50, 100, 200, 400,800, 1000];
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(contextDictionarySize);

accuracy_results = cell(length(contextDictionarySize)+1, length(neighbourhoodSize)*length(keypointDictionarySize)+1);
config_table = zeros(n_sz*k_sz*c_sz, 3);
idx=1;
for k = 1: length(contextDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Context', contextDictionarySize(k));
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, (i-1)*length(keypointDictionarySize)+j+1} = sprintf('%dN %d_KD', neighbourhoodSize(i), keypointDictionarySize(j));
            config_table(idx,:) = [contextDictionarySize(k),neighbourhoodSize(i), keypointDictionarySize(j)];
            idx= idx+1;
          end      
     end
end

%%


acc_log = cell(1, size(config_table, 1));

parfor i = 1: size(config_table, 1)
            
            inFName = fullfile(data_dir, sprintf('3Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(i,3),config_table(i,2)), sprintf('_soft_texton_ind_%d_%d_neigh_contexthist%d', config_table(i,1), config_table(i,2), coarseDictionarySize)));
	    try
		    H_all = load(inFName,'-ascii');
		    dat = load(train_test_datafilename);
		    train = dat.train;
		    test = dat.test;
		    train_features = H_all(train.index,:);
		    test_features = H_all(test.index,:);

		    %We'll do multiclass classification
		    %Compute kernel for training data
		    %train.K = [(1:size(train_features,1))' , pdist2(train_features, train_features)];
		    %test.K = [(1:size(test_features,1))' , pdist2(test_features, train_features)];
	    
		    bow_model = svmtrain(train.labels',train.features, '-t 0' );
		    [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels',test.features, bow_model);
		    acc_log{i} = accuracy_P(1);
 	   catch 
		    acc_log{i} = '';
 	   end 
end

idx=1;
for k = 1: length(contextDictionarySize)
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{k+1, (i-1)*length(keypointDictionarySize)+j+1} = acc_log(idx);
            idx= idx+1;
          end      
     end
end



outfname = fullfile( data_dir, 'results', '3DVLADContextHist_accuracy.mat');
save(outfname, 'accuracy_results');












