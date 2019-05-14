clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/Direc';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%% Build features using appended half and quarter context histograms on the go
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
%Histograms of keypoint sift with directional orientation context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ 40;
                       80;
                       ];
keypointDictionarySize = [50, 100, 200, 400,800, 1000];

accuracy_results = cell(length(neighbourhoodSize)*2+1, length(keypointDictionarySize)+1);
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);

config_table = zeros(2*n_sz*k_sz, 3);
idx=1;
for k = 1: 2
     for i = 1:length(neighbourhoodSize)
      if(k==1)
       accuracy_results{2*(k-1)+i+1, 1} = sprintf('%d_NQua', neighbourhoodSize(i));
      else
       accuracy_results{2*(k-1)+i+1, 1} = sprintf('%d_NHal', neighbourhoodSize(i));
      end
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, j+1} = sprintf('%d_KD', keypointDictionarySize(j));
            config_table(idx,:) = [k, neighbourhoodSize(i), keypointDictionarySize(j)];
            idx= idx+1;
          end      
     end
end

%%

acc_log = cell(1, size(config_table, 1));
parfor i = 1: size(config_table, 1)
            
	    if(config_table(i,1)==1)
	        pool_reg = 'q';
	    else
		pool_reg = 'h'
	    end
	    H_all = [];
	    for k = 1:4
            inFName = fullfile(data_dir, sprintf('soft_histograms_%d%s', config_table(i,3), sprintf('_%d_%s%d_neigh_ROcontext_keypoint_sift.mat', config_table(i,2), pool_reg, k )));
            h = load(inFName,'-ascii');
            H_all = [H_all, h];
	    end
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification with decision trees
    
%            tree_model = fitctree(train_features, train.labels', 'SplitCriterion', 'deviance');
            tree_model = fitensemble(train_features, train.labels', 'AdaBoostM2', 200, 'Tree');
	    test_predict = predict(tree_model,test_features);
            acc_log{i} = 100*sum(test_predict==test.labels',1)/length(test.labels);
end


idx=1;
for k = 1:2 
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{2*(k-1)+i+1, j+1} = acc_log{idx};
            idx= idx+1;
          end      
     end
end



outfname = fullfile( data_dir, 'results', 'directionalRO_tree_accuracy.mat');
save(outfname, 'accuracy_results');
