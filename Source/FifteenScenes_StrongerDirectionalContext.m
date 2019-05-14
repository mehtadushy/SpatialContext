% Directionally computes 'Stronger Context' in the neighbourhood of keypoints
% by considering a histogram of relative orientations of the keypoints in the 
% neighbourhood, considered per sector, as well as a coarse histogram of keypoint
% SIFT computed at those keypoints. The 'Stronger Context' is then clustered
% independently per direction and 2D histogram constructed for an image, with
% appearance captured by sift of the central keypoint along one dimension and 
% context along the other direction.
% The features are then weighted by tf-idf before being used for classification with
% SVMs
clear;
%image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/StrongerContext';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
%load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'rot3_scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Flags
DoG_keypoints = false; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

canSkip_FrameComputation = 0;
params.axisAlignedSIFT = false;
%Compute keypoint sift frames
if(DoG_keypoints==true)
    GenerateDoGKeypointSiftDescriptors( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,[]);
else
    GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,[]);
end

%% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [100, 200, 400, 800];
keypointSuffix = '_keypoint_sift.mat';
parfor k = 1:length(dictionarySize)
	params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
	CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
	BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
end

%% Compute context features
%and also sneakily compute relative orientation information while at it
%Compute relative orientation context of keypoint and coarse keypoint context
params.sigmaFactors = [%40;
                       80;
                      ];
params.keypointFeatureSuffix = '_keypoint_sift.mat';
params.completeNeighbourhood = false;
params.halfNeighbourhood = true;
params.quarterNeighbourhood = true;
params.keypointHistogramSize = [0;
                               %40;
                               ];
ComputeKeypointNeighbourhood(data.filenames, data_dir, params, 0);
% Go about clustering the context features
sigmaFactors = params.sigmaFactors;
keypointHistogramSize = params.keypointHistogramSize;
canSkip_SIFTComputation = 0;

contextFeature_table = cell(length(sigmaFactors)*8*length(keypointHistogramSize),1);
contextFeature_param = zeros(length(sigmaFactors)*8*length(keypointHistogramSize),3);
%
idx = 1;
for cd = 1: length(keypointHistogramSize)
    for sf = 1:length(sigmaFactors)
        for h = 1:4
        contextFeature_table{idx} = sprintf('_%dH_%dN_%dCD_ro_context', h, sigmaFactors(sf), keypointHistogramSize(cd));
        contextFeature_param(idx,:) = [h, sigmaFactors(sf), keypointHistogramSize(cd)];
        idx = idx+1;
        end
        
        for q = 1:4
        contextFeature_table{idx} = sprintf('_%dQ_%dN_%dCD_ro_context', q, sigmaFactors(sf), keypointHistogramSize(cd));
        contextFeature_param(idx,:) = [q, sigmaFactors(sf), keypointHistogramSize(cd)];
        idx = idx+1;
        end
    end
end
%


%
% Cluster Keypoint Context
neighbourhoodSize = params.sigmaFactors;
coarseDicSize = params.keypointHistogramSize;
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
contextDictionarySize = [10, 20];%, 80 ,100];
parfor i = 1:size(contextFeature_table,1)
    for k = 1:length(contextDictionarySize)
            para = struct('numTextonImages', 1000, 'dictionarySize', contextDictionarySize(k));
        %    params.dictionarySize = dictionarySize(k);
            CalculateHistogramDictionary(data.filenames,data_dir,sprintf('%s.mat',contextFeature_table{i}),para, 0 ,[]);
            BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,sprintf('%s.mat',contextFeature_table{i}),para,0, []);
    end
    
end


% Next up, we would construct 2D Histograms 

%contextDictionarySize = [40, 80 ,100];
keypointDictionarySize = [100, 200, 400,800];

parfor i = 1:size(contextFeature_table,1)
for j = 1: length(keypointDictionarySize)
    for k = 1: length(contextDictionarySize)

            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_keypoint_sift',keypointDictionarySize(j)), sprintf('_soft_texton_ind_%d%s',contextDictionarySize(k), contextFeature_table{i}), [keypointDictionarySize(j), contextDictionarySize(k)],[] , 0 );

    end
end
end

%
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
% train_percent = 90;
% train.index = (1 : floor(length(data.labels)*train_percent/100));
% train.labels = data.labels(train.index);
% test.index = (length(train.index)+1 : length(data.labels));
% test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');


%
%%%%%%%%%%%%%%%%
%Histograms of keypoint sift with directional orientation context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ %40;
                       80;
                       ];
keypointDictionarySize = [100, 200, 400,800];
%contextDictionarySize = [40, 80 ,100];

accuracy_results = cell(length(neighbourhoodSize)*2+1, length(contextDictionarySize)* length(keypointDictionarySize)+1);
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(contextDictionarySize);

config_table = zeros(2*n_sz*k_sz*c_sz, 4);
idx=1;
for k = 1: 2
     for i = 1:length(neighbourhoodSize)
      if(k==1)
       accuracy_results{n_sz*(k-1)+i+1, 1} = sprintf('%d_NQua', neighbourhoodSize(i));
      else
       accuracy_results{n_sz*(k-1)+i+1, 1} = sprintf('%d_NHal', neighbourhoodSize(i));
      end
          for j = 1: length(keypointDictionarySize)
              for l = 1:c_sz
                accuracy_results{1, c_sz*(j-1)+l+1} = sprintf('%d_KD %d_CD', keypointDictionarySize(j), contextDictionarySize(l));
                config_table(idx,:) = [k, neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(l)];
                idx= idx+1;
              end
          end      
     end
end

%%
classes = unique(train.labels);
acc_log = cell(1, size(config_table, 1));
parfor i = 1: size(config_table, 1)
            
	    if(config_table(i,1)==1)
	        pool_reg = 'Q';
        else
            pool_reg = 'H'
	    end
	    H_all = [];
	    for k = 1:4
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_keypoint_sift',config_table(i,3)), sprintf('_soft_texton_ind_%d_%d%s_%dN_%dCD_ro_context', config_table(i,4),k, pool_reg, config_table(i,2), 0))); 
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
             fprintf('C = %d\n',C);
             params = sprintf('-q -c %d -t 4', C);
             bow_model = svmtrain(train.labels',train.K, params );
             [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
             acc_log{i} = accuracy_P(1);
            
end


idx=1;
for k = 1:2 
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
               for l = 1: c_sz
                 accuracy_results{2*(k-1)+i+1, c_sz*(j-1)+l+1} = acc_log{idx};
                idx= idx+1;
               end
          end      
     end
end



outfname = fullfile( data_dir, 'results', 'directionalStrong_accuracy.mat');
save(outfname, 'accuracy_results');


