% Computes a stronger(than just relative orientation) context that captures
% both relative orientation and types of SIFTs occurring in the
% neighbourhood of a keypoint through a 2D histogram. The relative
% orientation is pushed into the keypoint orientation, while the coarse
% sift histogram from the neighbourhood is summed/averaged per keypoint cluster. 

clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Flags
DoG_keypoints = true; % uses DoG keypoints for SIFT
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
params.sigmaFactors = [%40;
                       80;
                       ];
params.keypointFrameSuffix = '_keypoint_sift_frame.mat';

ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, canSkip_FrameComputation);

% Now use the frames to compute Keypoint SIFT
sigmaFactors = params.sigmaFactors;

canSkip_SIFTComputation = 0;

for sf = 1:length(sigmaFactors)
    %clear params;
    params.keypointFramePrefix = sprintf('_%d_neigh_ROcontext', sigmaFactors(sf));
    GenerateKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
end


%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [%20, 40, 50, 100, 
                 20 , 50, 100, 200, 400, 800, 1000];
sigmaFactors = [%40;
		80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
    parfor k = 1:length(dictionarySize)
        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
    end
end
%%
clear params;
% Create coarse SIFT histograms in keypoint neighbourhoods for use as context
canSkip_ContextComputation = 0;
coarseDictionarySize = [20, 40, 80]; %Coarse Dictionary Size
sigmaFactors = [%40;
		80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
    parfor k = 1:length(coarseDictionarySize)
   	textonSuffix = sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift.mat', coarseDictionarySize(k), sigmaFactors(sf));
        params = struct('dictionarySize', coarseDictionarySize(k),'keypointFeatureSuffix',keypointSuffix, 'sigmaFactor', sigmaFactors(sf), 'scaleWeighted', false);
        ComputeCoarseNeighbourhoodHistogram(data.filenames,data_dir,textonSuffix, params, canSkip_ContextComputation);
    end
end
clear params;


%%
% Collect the coarse histogram context per keypoint cluster by summing or averaging per cluster.
%

neighbourhoodSize = [ %40;
                       80;
                       ];
%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
keypointDictionarySize = [100,200, 400, 800 , 1000
            ];
parfor j = 1: length(keypointDictionarySize)
    for l = 1: length(coarseDictionarySize)
      for i = 1:length(neighbourhoodSize)
          params = struct('clusterWeighted', false);
          CompileRawNeighbourhood2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',keypointDictionarySize(j),neighbourhoodSize(i)), sprintf('_%d_neigh_contexthist%d', neighbourhoodSize(i), coarseDictionarySize(l)), [keypointDictionarySize(j), coarseDictionarySize(l)],params , 0 );
     end
    end
%end


load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

%
%%%%%%%%%%%%%%%%
% 2D Histograms of keypoint appearance captured with SIFT and keypoint
% context
%%%%%%%%%%%%%%%%
n_sz = length(neighbourhoodSize);
k_sz = length(keypointDictionarySize);
c_sz = length(coarseDictionarySize);

accuracy_results = cell(length(coarseDictionarySize)+1, length(neighbourhoodSize)*length(keypointDictionarySize)+1);
config_table = zeros(n_sz*k_sz*c_sz, 3);
idx=1;
for k = 1: length(coarseDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Coarse', coarseDictionarySize(k));
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results{1, (i-1)*length(keypointDictionarySize)+j+1} = sprintf('%dN %d_KD', neighbourhoodSize(i), keypointDictionarySize(j));
            config_table(idx,:) = [coarseDictionarySize(k),neighbourhoodSize(i), keypointDictionarySize(j)];
            idx= idx+1;
          end      
     end
end

%


classes = unique(train.labels);
acc_log = cell(1, size(config_table, 1));

parfor i = 1: size(config_table, 1)
            
            inFName = fullfile(data_dir, sprintf('2Dhistograms_%s_with%s.mat', sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',config_table(i,3), config_table(i,2)), sprintf('_%d_neigh_contexthist%d', config_table(i,2), config_table(i,1))));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

%          sqrt_features = sqrt(train_features);
%           % Compute feature density ratios of one class vs rest and use
%           % the most distinctive features for classification with svm
%           feat_idx = [];
%           mean_c = zeros(length(classes), size(train_features,2));
%
%           for c = 1: length(classes)
%               class = classes(c);
%               mean_c(c,:) = mean(sqrt_features(train.labels == class,:));
%           end
%          for c = 1: length(classes)
%                complement_mean = mean(mean_c( ~((1:length(classes)) == c) , :));
%                density = mean_c(c,:) ./ complement_mean;
%                imp = find(density > 1);
%                feat_idx = [feat_idx find(imp>0)];  
%          end
%           feat_idx = unique(feat_idx);
%
%            %We'll do multiclass classification
%            %Compute kernel for training data
%             train.K = [(1:size(train_features,1))' , intersectionKernel(train_features(:,feat_idx), train_features(:,feat_idx))];
%             test.K = [(1:size(test_features,1))' , intersectionKernel(test_features(:,feat_idx), train_features(:,feat_idx))];
             train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
             test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
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
end

idx=1;
for k = 1: length(coarseDictionarySize)
     for i = 1:length(neighbourhoodSize)
          for j = 1: length(keypointDictionarySize)
            accuracy_results(k+1, (i-1)*length(keypointDictionarySize)+j+1) = acc_log(idx);
            idx= idx+1;
          end      
     end
end



outfname = fullfile( data_dir, 'results', 'Raw2DContextHist_accuracy.mat');
save(outfname, 'accuracy_results');

