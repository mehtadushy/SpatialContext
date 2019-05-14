% Computes relative orientation context on a directional basis, 
% which is pushed into the keypoint orientation. The directional
% context altered SIFTs are then clustered independently and the
% histograms appended to create the feature histogram

clear;
%image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/newDir';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
%load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'rot3_scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Flags
DoG_keypoints = false; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

canSkip_FrameComputation = 0;
params.axisAlignedSIFT = false;
%Compute keypoint sift frames
pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Frames'));
if(DoG_keypoints==true)
    GenerateDoGKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
else
    GenerateKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
end
close(pfig);

%Compute relative orientation context of keypoint and push back into the keypoint orientation
params.sigmaFactors = [40;
                       80;
                       ];
params.keypointFrameSuffix = '_keypoint_sift_frame.mat';
params.completeNeighbourhood = false;
params.halfNeighbourhood = true;
params.quarterNeighbourhood = true;
ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, 0);

% Now use the frames to compute Keypoint SIFT
sigmaFactors = params.sigmaFactors;

canSkip_SIFTComputation = 0;

keypointFramePrefix_table = cell(length(sigmaFactors)*8,1);
%%
for sf = 1:length(sigmaFactors)
    for h = 1:4
    keypointFramePrefix_table{4*(sf-1)+h} = sprintf('_%d_h%d_neigh_ROcontext', sigmaFactors(sf), h);
    end
end
for sf = 1:length(sigmaFactors)
    for q = 1:4
    keypointFramePrefix_table{length(sigmaFactors)*4 + 4*(sf-1)+q} = sprintf('_%d_q%d_neigh_ROcontext', sigmaFactors(sf), q);
    end
end
%
parfor i = 1:size(keypointFramePrefix_table,1)
    params = struct('keypointFramePrefix', keypointFramePrefix_table{i,1}); 
    GenerateKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
end

%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [200, 400, 800, 1000];
parfor i = 1:size(keypointFramePrefix_table,1)
   keypointSuffix = sprintf('%s_keypoint_sift.mat', keypointFramePrefix_table{i,1});
    for k = 1:length(dictionarySize)
        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
    end
end

%% Build features using appended half and quarter context histograms on the go
load(fullfile(data_dir2, 'rot3_category_names.mat'));
%Do a test-train split (data is already randomized
%train_percent = 90;
%train.index = (1 : floor(length(data.labels)*train_percent/100));
%train.labels = data.labels(train.index);
%test.index = (length(train.index)+1 : length(data.labels));
%test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

%%
%%%%%%%%%%%%%%%%
%Histograms of keypoint sift with directional orientation context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ 40;
                       80;
                       ];
keypointDictionarySize = [200, 400,800, 1000];

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
classes = unique(train.labels);
acc_log = cell(1, size(config_table, 1));
%%
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
%             idf = log(size(train_features,1)./ (sum(train_features>0,1)+1));
%             train_features(train_features~=0) = 1+ log( train_features(train_features~=0) )
%             test_features(test_features~=0) = 1+ log( test_features(test_features~=0) )
%             train_features = bsxfun( @(A,B) A.*B, idf, train_features);
%             test_features = bsxfun( @(A,B) A.*B, idf, test_features);
%             train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
%             test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
          sqrt_features = sqrt(train_features);
           % Compute feature density ratios of one class vs rest and use
           % the most distinctive features for classification with svm
           feat_idx = [];
            mean_c = zeros(length(classes), size(train_features,2));

           for c = 1: length(classes)
               class = classes(c);
               mean_c(c,:) = mean(sqrt_features(train.labels == class,:));
           end
          for c = 1: length(classes)
                complement_mean = mean(mean_c( ~((1:length(classes)) == c) , :));
                density = mean_c(c,:) ./ complement_mean;
                imp = find(density > 1);
                feat_idx = [feat_idx find(imp>0)];  
          end
           feat_idx = unique(feat_idx);
           
%             for c = 1: length(classes)
%                 class = classes(c);
%                 feat1 = sqrt_features(train.labels == class,:);
%                 feat2 = sqrt_features(train.labels ~= class,:);
%                 density = mean(feat1)./mean(feat2);
%                 imp = find(density > 1)
%                 feat_idx = [feat_idx find(imp>0)];                
%             end
   
%            feat_idx = unique(feat_idx);

            %We'll do multiclass classification
            %Compute kernel for training data
             train.K = [(1:size(train_features,1))' , intersectionKernel(train_features(:,feat_idx), train_features(:,feat_idx))];
             test.K = [(1:size(test_features,1))' , intersectionKernel(test_features(:,feat_idx), train_features(:,feat_idx))];
                 
            bestcv = 0;
            for log2c = -2:2:4
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; bestc = 2^log2c; bestlog2c = log2c;
                end
            end
            fprintf('Best C = %d\n',bestc);
            bestcv = 0;
            for log2c = (bestlog2c-0.3):0.1:(bestlog2c+0.3)
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; C = 2^log2c;
                end
            end
             fprintf('C = %d\n',bestc);
             params = sprintf('-q -c %d -t 4', C);
             bow_model = svmtrain(train.labels',train.K, params );
             [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
             acc_log{i} = accuracy_P(1);
                        % fprintf('Features Selected %d\n',length(feat_idx));

            
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



outfname = fullfile( data_dir, 'results', 'directionalRO_accuracy.mat');
save(outfname, 'accuracy_results');
accuracy_results
%%
% classes = unique(train.labels);
% acc_log = cell(1, size(config_table, 1));
% parfor i = 1: size(config_table, 1)
%             
% 	    if(config_table(i,1)==1)
% 	        pool_reg = 'q';
% 	    else
% 		pool_reg = 'h'
% 	    end
% 	    H_all = [];
% 	    for k = 1:4
%             inFName = fullfile(data_dir, sprintf('soft_histograms_%d%s', config_table(i,3), sprintf('_%d_%s%d_neigh_ROcontext_keypoint_sift.mat', config_table(i,2), pool_reg, k )));
%             h = load(inFName,'-ascii');
%             H_all = [H_all, h];
% 	    end
%             dat = load(train_test_datafilename);
%             train = dat.train;
%             test = dat.test;
%             train_features = H_all(train.index,:);
%             test_features = H_all(test.index,:);
%             
%             % First train a bunch of random forests to extract out the
%             % useful features for one-vs-all classification, and combine
%             % the extracted features for all classes. Then use this subset
%             % of features for training svm
%             feat_idx = [];
%             for c = 1: length(classes)
%                 class = classes(c);
%                 tempTrainLabels = train.labels;
%                 tempTrainLabels(train.labels == class) = 1; 
%                 tempTrainLabels(train.labels ~= class) = 2;
%                 tree_model = fitensemble(train_features, tempTrainLabels', 'AdaBoostM1', 20, 'Tree');
%                 imp = predictorImportance(tree_model);
%                 feat_idx = [feat_idx find(imp>0)];                
%             end
%             
%             feat_idx = unique(feat_idx);
% 
%             %We'll do multiclass classification
%             %Compute kernel for training data
%              train.K = [(1:size(train_features,1))' , intersectionKernel(train_features(:,feat_idx), train_features(:,feat_idx))];
%              test.K = [(1:size(test_features,1))' , intersectionKernel(test_features(:,feat_idx), train_features(:,feat_idx))];
%      
%              bow_model = svmtrain(train.labels',train.K, '-t 4' );
%              [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
%              acc_log{i} = accuracy_P(1);
%             
% end
% 
% 
% idx=1;
% for k = 1:2 
%      for i = 1:length(neighbourhoodSize)
%           for j = 1: length(keypointDictionarySize)
%             accuracy_results{2*(k-1)+i+1, j+1} = acc_log{idx};
%             idx= idx+1;
%           end      
%      end
% end
% 
% 
% 
% outfname = fullfile( data_dir, 'results', 'directionalRO_accuracy.mat');
% save(outfname, 'accuracy_results');













